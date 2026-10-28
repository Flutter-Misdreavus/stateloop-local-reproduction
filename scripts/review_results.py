"""Independent completion audit; refuses to certify incomplete experiment sets.

Recomputes token-weighted NLL/PPL and paired document-bootstrap intervals from
local per-document measurements. This produces evidence for human/GPT review,
not an automatic scientific endorsement of the mechanism.
"""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from datetime import datetime
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from stateloop.runtime import digest,dump


ROOT=Path("experiments/stateloop-20261028-local48")
EXPECTED={(v,s) for v in ["vanilla","reset","persistent"] for s in [17,42]}


def close(a,b,atol=1e-9):
    assert math.isfinite(a) and math.isfinite(b)
    assert abs(a-b)<=atol, (a,b)


def metric_check(metric):
    assert metric["tokens"]>0
    close(metric["nll"],metric["nll_sum"]/metric["tokens"])
    close(metric["ppl"],math.exp(metric["nll"]),atol=1e-7)


def core_hash(revision):
    h=hashlib.sha256()
    for filename in ["stateloop/model.py","stateloop/train.py","stateloop/runtime.py"]:
        h.update(filename.encode())
        h.update(subprocess.check_output(["git","show",f"{revision}:{filename}"]))
    return h.hexdigest()


def independent_bootstrap(a,b):
    keys=sorted(a["documents"])
    assert keys==sorted(b["documents"])
    loss_diff=np.asarray([a["documents"][k][0]-b["documents"][k][0] for k in keys])
    counts=np.asarray([a["documents"][k][1] for k in keys],dtype=np.float64)
    assert all(a["documents"][k][1]==b["documents"][k][1] for k in keys)
    rng=np.random.default_rng(20261028)
    values=[]
    for _ in range(2000):
        weights=np.bincount(rng.integers(0,len(keys),len(keys)),minlength=len(keys))
        values.append(float(weights@loss_diff/(weights@counts)))
    return np.quantile(values,[.025,.975]).tolist()


def main():
    torch.set_num_threads(4)
    matrix=json.load(open(ROOT/"matrix.json"))
    assessment=json.load(open(ROOT/"assessment.json"))
    frozen=json.load(open("configs/frozen.json"))
    manifest=json.load(open("runtime/data/manifest.json"))
    deadline=datetime.fromisoformat(frozen["deadline_utc"].replace("Z","+00:00")).timestamp()
    assert matrix["status"]=="SUCCEEDED_PENDING_REVIEW", "matrix is not complete"
    assert assessment["status"]=="SUCCEEDED", "assessment is not complete"
    assert assessment["ended_epoch"]<=deadline
    assert {(r["variant"],r["seed"]) for r in matrix["runs"]}==EXPECTED
    assert len(matrix["runs"])==6 and len(assessment["runs"])==6
    assert digest("runtime/data/manifest.json")==frozen["data_manifest_sha256"]
    for name,info in manifest["artifacts"].items():
        assert digest(Path("runtime/data")/name)==info["sha256"]
    report={"status":"RUNNING","review_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"assessment_sha256":digest(ROOT/"assessment.json"),"frozen_config_sha256":digest("configs/frozen.json"),"runs":[],"recomputed_pairs":[],"scientific_review":"PENDING"}
    documents={}
    for entry in matrix["runs"]:
        run_id=entry["run_id"]
        run=json.load(open(Path("experiments")/run_id/"report.json"))
        key=(run["variant"],run["seed"])
        assert key==(entry["variant"],entry["seed"])
        assert run["status"]=="SUCCEEDED" and run["reliability"]=="MET" and not run["failures"]
        assert run["ended_or_updated_epoch"]<=deadline
        assert run["tokens"]==run["target_tokens"]==frozen["tokens_per_run"]
        assert run["steps"]*frozen["batch_sequences"]*frozen["model"]["seq_len"]==run["tokens"]
        assert run["frozen_config_sha256"]==report["frozen_config_sha256"]
        assert run["data_manifest_sha256"]==frozen["data_manifest_sha256"]
        assert {k:v for k,v in run["config"].items() if k!="variant"}==frozen["model"]
        assert entry["run_commit"]==run["run_commit"]
        metric_check(run["initial_dev"])
        metric_check(run["final_dev"])
        close(run["dev_nll_reduction"],run["initial_dev"]["nll"]-run["final_dev"]["nll"])
        assert run["dev_nll_reduction"]>=.5
        checkpoint=run["checkpoint"]
        assert digest(checkpoint["path"])==checkpoint["sha256"]
        assert Path(checkpoint["path"]).stat().st_size==checkpoint["bytes"]
        saved=torch.load(checkpoint["path"],map_location="cpu")
        assert saved["tokens"]==run["tokens"] and saved["run_commit"]==run["run_commit"]
        assert saved["config"]==run["config"] and saved["seed"]==run["seed"]
        assert sum(t.numel() for t in saved["model"].values())==run["params"]
        assert run["params"]==(52_795_776 if run["variant"]=="vanilla" else 53_387_136)
        assert all(torch.isfinite(t).all().item() for t in saved["model"].values())
        del saved
        training=[json.loads(line) for line in (Path("experiments")/run_id/"training.jsonl").read_text().splitlines()]
        assert training[-1]["tokens"]==run["tokens"]
        assert all(a["step"]<b["step"] for a,b in zip(training,training[1:]))
        assert all(math.isfinite(row["train_nll"]) and math.isfinite(row["gradient_norm"]) for row in training)
        test=json.load(open(Path("experiments")/run_id/"test.json"))
        raw=json.load(open(Path("runtime/runs")/run_id/"test-document-nll.json"))
        documents[key]=raw
        metric_check(raw)
        metric_check(test)
        close(raw["nll"],test["nll"])
        assert test["tokens"]==manifest["artifacts"]["test.npz"]["tokens"]
        close(sum(x[0] for x in raw["documents"].values()),test["nll_sum"],atol=1e-7)
        assert sum(x[1] for x in raw["documents"].values())==test["tokens"]
        assert test["run_commit"]==run["run_commit"] and test["checkpoint_sha256"]==checkpoint["sha256"]
        summary=next(r for r in assessment["runs"] if (r["variant"],r["seed"])==key)
        close(summary["nll"],test["nll"])
        report["runs"].append({"run_id":run_id,"variant":key[0],"seed":key[1],"tokens":run["tokens"],"run_commit":run["run_commit"],"core_code_sha256":core_hash(run["run_commit"]),"checkpoint_sha256":checkpoint["sha256"],"test_nll":test["nll"],"test_ppl":test["ppl"],"training_elapsed_s":run["training_elapsed_s"],"invocation_elapsed_s":run["elapsed_s_this_invocation"],"peak_memory_bytes":run["peak_memory_bytes"]})
    assert len({r["core_code_sha256"] for r in report["runs"]})==1, "training code changed across paired runs"
    deltas=[]
    for seed in [17,42]:
        official=next(p for p in assessment["paired"] if p["seed"]==seed)
        pair={"seed":seed}
        for name,left in [("persistence","reset"),("total","vanilla")]:
            a,b=documents[(left,seed)],documents[("persistent",seed)]
            delta=a["nll"]-b["nll"]
            close(delta,official[name]["difference_nats"])
            interval=independent_bootstrap(a,b)
            for x,y in zip(interval,official[name]["document_bootstrap_95pct"]):
                close(x,y,atol=1e-10)
            pair[name]={"delta_nll":delta,"document_bootstrap_95pct":interval}
            if name=="persistence":
                deltas.append(delta)
        report["recomputed_pairs"].append(pair)
    expected_effects={"feedback_off","terminal_off","all_off"}|{f"{mode}_S{k}" for mode in ["single","frozen"] for k in range(1,4)}
    assert len(assessment["interventions"])==18
    for seed in [17,42]:
        calibration=json.load(open(Path("experiments")/f"stateloop-20261028-persistent-s{seed}"/"calibration.json"))
        assert calibration["positions"]==500_000
        for name in ["mean","std"]:
            values=np.asarray(calibration[name])
            assert values.shape==(4,384) and np.isfinite(values).all()
            if name=="std":
                assert (values>=0).all()
        effects={r["intervention"]:r for r in assessment["interventions"] if r["seed"]==seed}
        assert set(effects)==expected_effects
        for effect in effects.values():
            metric_check(effect)
            assert effect["tokens"]==1_000_000
            close(effect["delta_nll"],effect["nll"]-documents[("persistent",seed)]["nll"])
        close(effects["single_S3"]["nll"],effects["frozen_S3"]["nll"],atol=1e-7)
    mean=float(np.mean(deltas))
    support=all(d>0 for d in deltas) and mean>=.005
    close(mean,assessment["decision"]["mean_persistence_delta_nats"])
    assert support==assessment["decision"]["preliminary_support"]
    report.update(status="PASSED",mean_persistence_delta_nats=mean,preliminary_support=support,all_six_training_code_identical=True,total_formal_training_tokens=sum(r["tokens"] for r in report["runs"]),summed_training_hours=sum(r["training_elapsed_s"] for r in report["runs"])/3600,intervention_count=18,limitations=["two training seeds","approximately 100M tokens per run","scaled architecture and deduplicated Pile subset","document bootstrap is conditional on trained models","exact-overlap checks do not exclude all near duplicates"])
    dump(ROOT/"completion-audit.json",report)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
