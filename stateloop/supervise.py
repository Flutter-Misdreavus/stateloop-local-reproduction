"""Local sequential matrix, absolute timeout and sanitized progress publication.

This executes only the frozen protocol. It never changes scientific settings,
retries a failed scientific gate, or labels automated results GPT-reviewed.
"""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .runtime import clean_code, commit, deadline_epoch, dump

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path("experiments/stateloop-20261028-local48")


def publish(stage):
    # Allowlisted generated evidence only; raw runtime artifacts never staged.
    clean_code()
    paths = []
    for p in Path("experiments").rglob("*"):
        if p.is_file() and p.suffix in {".json", ".jsonl", ".md"} and p.stat().st_size < 5_000_000:
            paths.append(str(p))
    if paths:
        subprocess.run(["git", "add", "--", *paths], check=True)
    subprocess.run(["git", "diff", "--cached", "--check"], check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode:
        subprocess.run(["git", "commit", "-m", f"stateloop-20261028-local48 {stage}: publish aggregate execution evidence"], check=True)
    try:
        visibility = subprocess.check_output(["gh", "repo", "view", "Flutter-Misdreavus/stateloop-local-reproduction", "--json", "isPrivate,nameWithOwner"], text=True, timeout=30)
        info = json.loads(visibility)
        if info["isPrivate"] or info["nameWithOwner"] != "Flutter-Misdreavus/stateloop-local-reproduction":
            raise RuntimeError("Unexpected repository identity or visibility")
        result = subprocess.run(["git", "push", "origin", "main"], text=True, capture_output=True, timeout=60)
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        dump("runtime/push-status.json", {"status": "PUSHED", "commit": commit(), "epoch": time.time()})
    except Exception as e:
        dump("runtime/push-status.json", {"status": "PENDING", "commit": commit(), "epoch": time.time(), "error": str(e)})
        print("PUSH_PENDING: " + str(e), flush=True)


def run_child(command, log_path, deadline, stage):
    with open(log_path, "a") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        next_publish = time.time()+3600
        while child.poll() is None:
            if time.time() >= deadline:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                raise TimeoutError(stage + " exceeded registered deadline")
            if time.time() >= next_publish:
                publish("progress")
                next_publish = time.time()+3600
            dump("runtime/heartbeat.json", {"epoch": time.time(), "stage": stage, "child_pid": child.pid, "deadline": deadline, "note": "liveness only; consult actual training metrics"})
            time.sleep(15)
        if child.returncode:
            raise RuntimeError(f"{stage} exited with code {child.returncode}; no automatic scientific retry")


def main():
    os.chdir(ROOT)
    Path("runtime").mkdir(exist_ok=True)
    lock = open("runtime/supervisor.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    clean_code()
    frozen = json.load(open("configs/frozen.json"))
    if not frozen["ready"]:
        raise RuntimeError("Configuration not frozen")
    order = [("vanilla", 17), ("reset", 17), ("persistent", 17), ("persistent", 42), ("reset", 42), ("vanilla", 42)]
    matrix_path = EVIDENCE/"matrix.json"
    if matrix_path.exists():
        raise RuntimeError("Matrix already exists: inspect status; do not silently restart")
    start = time.time()
    training_deadline = min(start+34*3600, deadline_epoch()-4*3600)
    matrix = {"status": "RUNNING", "supervisor_commit": commit(), "started_epoch": start, "absolute_deadline_epoch": deadline_epoch(), "training_deadline_epoch": training_deadline, "runs": [{"variant": v, "seed": s, "run_id": f"stateloop-20261028-{v}-s{s}", "status": "PENDING"} for v,s in order]}
    dump(matrix_path, matrix)
    try:
        for run in matrix["runs"]:
            run["status"] = "RUNNING"
            dump(matrix_path, matrix)
            publish("stage-start")
            command = [sys.executable, "-m", "stateloop.train", "--variant", run["variant"], "--seed", str(run["seed"]), "--run-id", run["run_id"], "--wall-deadline", str(training_deadline)]
            run_child(command, f"runtime/{run['run_id']}.log", training_deadline, run["run_id"])
            report = json.load(open(Path("experiments")/run["run_id"]/"report.json"))
            if report["status"] != "SUCCEEDED" or report["reliability"] != "MET":
                raise RuntimeError("Formal gate not met")
            run.update(status="SUCCEEDED", run_commit=report["run_commit"])
            dump(matrix_path, matrix)
            publish("stage-complete")
        matrix["status"] = "EVALUATING"
        dump(matrix_path, matrix)
        run_child([sys.executable, "-m", "stateloop.assess"], "runtime/assessment.log", min(time.time()+4*3600, deadline_epoch()), "assessment")
        matrix["status"] = "SUCCEEDED_PENDING_REVIEW"
    except Exception as e:
        matrix.update(status="FAILED", error=f"{type(e).__name__}: {e}", review_required=True)
        for run in matrix["runs"]:
            if run["status"] == "RUNNING":
                run["status"] = "FAILED"
        raise
    finally:
        matrix["ended_or_updated_epoch"] = time.time()
        dump(matrix_path, matrix)
        publish("matrix-finished")


if __name__ == "__main__":
    main()
