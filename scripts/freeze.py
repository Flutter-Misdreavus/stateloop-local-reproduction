"""Freeze only predeclared settings after data and preflight gates pass."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stateloop.runtime import digest, dump


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "formal"], required=True)
    args = parser.parse_args()
    root = Path("experiments/stateloop-20261028-local48")
    benchmark = json.load(open(root/"benchmark-preferred.json"))
    overfit = json.load(open(root/"overfit-preferred.json"))
    if benchmark["status"] != "PASSED" or overfit["status"] != "PASSED":
        raise RuntimeError("Preflight gates are not met")
    manifest = json.load(open("runtime/data/manifest.json"))
    for name, info in manifest["artifacts"].items():
        if digest(Path("runtime/data")/name) != info["sha256"]:
            raise RuntimeError("Prepared data integrity failure: "+name)
    protocol = json.load(open("configs/protocol.json"))
    slowest = min(r["tokens_per_second"] for r in benchmark["records"])
    target = min(100_000_000, int(.85*34*3600*slowest/6))//16384*16384
    if target < 50_000_000:
        raise RuntimeError("Preferred configuration does not meet minimum budget; re-review fallback")
    model = {**protocol["models"]["preferred"], "vocab_size": manifest["model_vocab_size"], "loops":4, "seq_len":512}
    cfg = {"ready":True, "stage":args.stage, "model":model, "batch_sequences":32, "micro_batch":8, "tokens_per_run":524288 if args.stage=="smoke" else target, "data_manifest_sha256":digest("runtime/data/manifest.json"), "benchmark_sha256":digest(root/"benchmark-preferred.json"), "overfit_sha256":digest(root/"overfit-preferred.json"), "freeze_script_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(), "minimum_formal_tokens":50_000_000, "minimum_mean_persistence_effect_nats":.005, "deadline_utc":protocol["deadline_utc"], "seeds":[123] if args.stage=="smoke" else [17,42]}
    if args.stage=="formal":
        smoke = json.load(open("experiments/stateloop-20261028-smoke/report.json"))
        if smoke["status"]!="SUCCEEDED" or smoke["reliability"]!="MET":
            raise RuntimeError("Real-data trainer smoke has not passed")
        cfg["smoke_report_sha256"] = digest("experiments/stateloop-20261028-smoke/report.json")
        protocol.update(ready=True,status="FROZEN")
        protocol["dataset"].update(revision=manifest["revision"],tokenizer_revision=manifest["tokenizer_revision"],manifest_sha256=cfg["data_manifest_sha256"])
        protocol["training"]["tokens"] = target
        protocol["models"]["selected"] = "preferred"
        dump("configs/protocol.json",protocol)
    path = "configs/smoke.json" if args.stage=="smoke" else "configs/frozen.json"
    dump(path,cfg)
    print(json.dumps(cfg,indent=2))


if __name__=="__main__":
    main()
