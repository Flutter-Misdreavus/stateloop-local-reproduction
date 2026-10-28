"""Locked-test evaluation, paired document bootstrap, and paper interventions."""
import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from .model import Config, StateLoopLM
from .runtime import clean_code, commit, deadline_epoch, digest, dump, evaluate


@torch.no_grad()
def calibration(model, device, micro):
    data = np.load("runtime/data/calibration.npz")
    counts = 0
    total = torch.zeros(model.cfg.loops, model.cfg.state_width, device=device, dtype=torch.float64)
    total2 = torch.zeros_like(total)
    for i in range(0, len(data["tokens"]), micro):
        if time.time() >= deadline_epoch():
            raise TimeoutError("Absolute deadline in calibration")
        x = torch.from_numpy(data["tokens"][i:i+micro, :-1].astype(np.int64)).to(device)
        lengths = torch.from_numpy(data["lengths"][i:i+micro].astype(np.int64)).to(device)
        mask = torch.arange(x.shape[1], device=device)[None] < lengths[:, None]
        with torch.autocast("cuda", dtype=torch.float16):
            _, states = model.features(x, return_states=True)
        for k, state in enumerate(states):
            v = state[mask].double()
            total[k] += v.sum(0)
            total2[k] += v.square().sum(0)
        counts += int(mask.sum())
    mean = total/counts
    std = (total2/counts-mean.square()).clamp_min(0).sqrt()
    return {"mean": mean.float(), "std": std.float(), "positions": counts}


def bootstrap_difference(left, right, seed=20261028):
    keys = sorted(left["documents"])
    assert keys == sorted(right["documents"])
    a = np.asarray([left["documents"][k] for k in keys], dtype=np.float64)
    b = np.asarray([right["documents"][k] for k in keys], dtype=np.float64)
    assert np.array_equal(a[:, 1], b[:, 1])
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(2000):
        ix = rng.integers(0, len(keys), len(keys))
        differences.append(float((a[ix, 0]-b[ix, 0]).sum()/a[ix, 1].sum()))
    return {"difference_nats": left["nll"]-right["nll"], "document_bootstrap_95pct": np.quantile(differences, [.025, .975]).tolist(), "documents": len(keys), "replicates": 2000, "scope": "evaluation-document uncertainty conditional on one trained model pair; not training-seed uncertainty"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="experiments/stateloop-20261028-local48/matrix.json")
    args = parser.parse_args()
    clean_code()
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    frozen = json.load(open("configs/frozen.json"))
    matrix = json.load(open(args.matrix))
    root = Path("experiments/stateloop-20261028-local48")
    output = {"status": "RUNNING", "evaluation_commit": commit(), "started_epoch": time.time(), "runs": [], "paired": [], "interventions": []}
    dump(root/"assessment.json", output)
    results = {}
    try:
        for run in matrix["runs"]:
            rep = json.load(open(Path("experiments")/run["run_id"]/"report.json"))
            if rep["status"] != "SUCCEEDED" or rep["reliability"] != "MET":
                raise RuntimeError("Refusing test evaluation before all formal runs meet reliability gates")
        for run in matrix["runs"]:
            rep = json.load(open(Path("experiments")/run["run_id"]/"report.json"))
            path = rep["checkpoint"]["path"]
            if digest(path) != rep["checkpoint"]["sha256"]:
                raise RuntimeError("Checkpoint hash mismatch")
            state = torch.load(path, map_location="cpu")
            model = StateLoopLM(Config(**state["config"]), seed=state["seed"]).to(device).eval()
            model.load_state_dict(state["model"])
            del state
            metrics = evaluate(model, "runtime/data/test.npz", device, micro_batch=frozen["micro_batch"], include_documents=True)
            results[(run["variant"], run["seed"])] = metrics
            dump(Path("runtime/runs")/run["run_id"]/"test-document-nll.json", metrics)
            summary = {k: v for k, v in metrics.items() if k != "documents"}
            summary.update(run_id=run["run_id"], variant=run["variant"], seed=run["seed"], run_commit=rep["run_commit"], evaluation_commit=output["evaluation_commit"], checkpoint_sha256=rep["checkpoint"]["sha256"], documents=len(metrics["documents"]))
            dump(Path("experiments")/run["run_id"]/"test.json", summary)
            output["runs"].append(summary)
            print(json.dumps(summary), flush=True)
            if run["variant"] == "persistent":
                stats = calibration(model, device, frozen["micro_batch"])
                dump(Path("experiments")/run["run_id"]/"calibration.json", {"mean": stats["mean"].cpu().tolist(), "std": stats["std"].cpu().tolist(), "positions": stats["positions"]})
                interventions = [(name, {name: True}) for name in ["feedback_off", "terminal_off", "all_off"]]
                for retained in range(1, model.cfg.loops):
                    for mode in ["single", "frozen"]:
                        interventions.append((f"{mode}_S{retained}", {"retain_boundary": retained, "mode": mode, "stats": stats}))
                for name, intervention in interventions:
                    effect = evaluate(model, "runtime/data/test.npz", device, micro_batch=frozen["micro_batch"], intervention=intervention)
                    effect.update(run_id=run["run_id"], seed=run["seed"], intervention=name, delta_nll=effect["nll"]-metrics["nll"])
                    output["interventions"].append(effect)
                    dump(root/"assessment.json", output)
                    print(json.dumps(effect), flush=True)
                del stats
            dump(root/"assessment.json", output)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        for seed in [17, 42]:
            output["paired"].append({"seed": seed, "persistence": bootstrap_difference(results[("reset", seed)], results[("persistent", seed)]), "total": bootstrap_difference(results[("vanilla", seed)], results[("persistent", seed)])})
        delta = [p["persistence"]["difference_nats"] for p in output["paired"]]
        mean = float(np.mean(delta))
        output["decision"] = {"mean_persistence_delta_nats": mean, "both_positive": all(x > 0 for x in delta), "minimum_mean_effect": .005, "preliminary_support": all(x > 0 for x in delta) and mean >= .005, "interpretation": "scaled mechanism reproduction; two seeds and limited pretraining do not establish general validity or invalidity", "gpt_review": "PENDING"}
        output["status"] = "SUCCEEDED"
        lines = ["# StateLoop local results", "", "Automated measurements; scientific review pending.", "", "| Variant | Seed | Test NLL | Test PPL |", "|---|---:|---:|---:|"]
        for row in sorted(output["runs"], key=lambda r: (r["seed"], r["variant"])):
            lines.append(f"| {row['variant']} | {row['seed']} | {row['nll']:.6f} | {row['ppl']:.4f} |")
        lines += ["", f"Mean paired reset-minus-persistent NLL: {mean:.6f} nat/token.", "", "Document bootstrap intervals are conditional on each trained pair. See assessment.json for all interventions, run commits and limits. Failure to find an effect at this training scale does not invalidate the full-scale paper.", ""]
        (root/"results.md").write_text("\n".join(lines))
    except Exception as e:
        output.update(status="FAILED", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        output["ended_epoch"] = time.time()
        dump(root/"assessment.json", output)


if __name__ == "__main__":
    main()
