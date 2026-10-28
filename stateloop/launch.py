"""Finish registered preparation, smoke, freeze, then run the local matrix.

Gate decisions are mechanical applications of the reviewed protocol. This does
not claim to replace the researcher's scientific interpretation of results.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .runtime import clean_code, commit, deadline_epoch, dump
from .supervise import publish, run_child

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path("experiments/stateloop-20261028-local48")


def commit_paths(paths, message):
    subprocess.run(["git", "add", "--", *paths], check=True)
    subprocess.run(["git", "diff", "--cached", "--check"], check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode:
        subprocess.run(["git", "commit", "-m", message], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-pid", type=int, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    lock = open("runtime/launch.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    clean_code()
    preparation_deadline = deadline_epoch()-42*3600
    status = {"status": "PREPARING_DATA", "launch_commit": commit(), "started_epoch": time.time(), "data_execution_commit": "683d216", "preparation_pid": args.prepare_pid, "preparation_deadline_epoch": preparation_deadline, "absolute_deadline_epoch": deadline_epoch(), "scientific_review": "PENDING"}
    dump(EVIDENCE/"launch.json",status)
    try:
        proc = Path(f"/proc/{args.prepare_pid}")
        if proc.exists():
            if Path(os.readlink(proc/"cwd")).resolve()!=ROOT or b"scripts/prepare.py" not in (proc/"cmdline").read_bytes():
                raise RuntimeError("Preparation PID does not belong to this experiment")
        while not Path("runtime/data/manifest.json").exists():
            if time.time() >= preparation_deadline:
                if proc.exists():
                    children = subprocess.run(["pgrep","-P",str(args.prepare_pid)],capture_output=True,text=True).stdout.split()
                    for pid in children+[str(args.prepare_pid)]:
                        try:
                            os.kill(int(pid),signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                raise TimeoutError("Six-hour preparation budget exhausted")
            if not proc.exists():
                raise RuntimeError("Pinned data preparation exited without a valid manifest; inspect preserved local transfer log")
            status["last_checked_epoch"] = time.time()
            dump(EVIDENCE/"launch.json",status)
            time.sleep(30)
        status["status"]="DATA_INTEGRITY_CHECK"
        dump(EVIDENCE/"launch.json",status)
        subprocess.run([sys.executable,"scripts/freeze.py","--stage","smoke"],check=True)
        commit_paths(["configs/smoke.json"],"stateloop-20261028-local48 smoke: freeze real-data integration check")
        publish("data-ready")
        status["status"]="REAL_DATA_SMOKE"
        dump(EVIDENCE/"launch.json",status)
        run_child([sys.executable,"-m","stateloop.train","--variant","persistent","--seed","123","--run-id","stateloop-20261028-smoke","--config","configs/smoke.json","--wall-deadline",str(preparation_deadline)],"runtime/smoke.log",preparation_deadline,"real-data-smoke")
        smoke=json.load(open("experiments/stateloop-20261028-smoke/report.json"))
        if smoke["status"]!="SUCCEEDED" or smoke["reliability"]!="MET":
            raise RuntimeError("Real-data integration gate did not pass")
        status["smoke_gate"]={"status":"MET","run_commit":smoke["run_commit"],"tokens":smoke["tokens"],"dev_nll_reduction":smoke["dev_nll_reduction"]}
        subprocess.run([sys.executable,"scripts/freeze.py","--stage","formal"],check=True)
        commit_paths(["configs/protocol.json","configs/frozen.json"],"stateloop-20261028-local48 formal: freeze paired six-run protocol after verified gates")
        status["status"]="FORMAL_MATRIX"
        dump(EVIDENCE/"launch.json",status)
        publish("formal-ready")
        run_child([sys.executable,"-m","stateloop.supervise"],"runtime/supervisor.log",deadline_epoch(),"formal-matrix",publish_progress=False)
        status["status"]="SUCCEEDED_PENDING_REVIEW"
    except Exception as e:
        status.update(status="FAILED",error=f"{type(e).__name__}: {e}",review_required=True)
        raise
    finally:
        status["ended_or_updated_epoch"]=time.time()
        dump(EVIDENCE/"launch.json",status)
        publish("launch-status")


if __name__=="__main__":
    main()
