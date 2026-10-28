"""Independent read-only audit of prepared data and full-size pairing.

Does not load trained models or inspect test-set quality. No text is exported.
"""
import hashlib
import json
from pathlib import Path
import sys
import unicodedata
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer
from stateloop.model import Config,StateLoopLM
from stateloop.runtime import commit,digest,dump


def partition(h):
    bucket=int(hashlib.sha256(b"stateloop-20261028:"+bytes.fromhex(h)).hexdigest()[:16],16)%1000
    return "train" if bucket<975 else "dev" if bucket<985 else "calibration" if bucket<990 else "test"


def tensor_hash(state,branch):
    result=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        if name.startswith("state.") != branch:
            continue
        result.update(name.encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def main():
    torch.set_num_threads(2)
    manifest=json.load(open("runtime/data/manifest.json"))
    frozen=json.load(open("configs/frozen.json"))
    tokenizer=Tokenizer.from_file("runtime/tokenizer/tokenizer.json")
    assert digest("runtime/data/manifest.json")==frozen["data_manifest_sha256"]
    assert digest("runtime/tokenizer/tokenizer.json")==manifest["tokenizer_sha256"]
    result={"audit_commit":commit(),"status":"RUNNING","split_checks":{},"sampled_document_checks":[],"pairing":[],"test_quality_inspected":False}
    pools, samples={},{}
    for split in ["dev","calibration","test"]:
        path=Path("runtime/data")/(split+".npz")
        assert digest(path)==manifest["artifacts"][path.name]["sha256"]
        data=np.load(path)
        ids=set(data["doc_ids"].tolist())
        assert all(partition(h)==split for h in ids)
        assert int(data["lengths"].sum())==manifest["artifacts"][path.name]["tokens"]
        assert data["tokens"].shape[1]==513
        assert int(data["tokens"].max())<manifest["vocab_size"]
        pools[split]=ids
        chosen=sorted(ids)[:2]+sorted(ids)[-2:]
        for h in chosen:
            ix=np.flatnonzero(data["doc_ids"]==h)
            samples[h]=(split,data["tokens"][ix].copy(),data["lengths"][ix].copy())
        result["split_checks"][split]={"documents":len(ids),"tokens":int(data["lengths"].sum()),"hash_partition_membership":True,"artifact_sha256_verified":True}
    assert not (pools["dev"]&pools["test"] or pools["dev"]&pools["calibration"] or pools["test"]&pools["calibration"])
    prefix,seen=[],set()
    needed=65536
    for source in manifest["source_files"]:
        path=Path("runtime/data/shards")/Path(source["file"]).name
        assert digest(path)==source["sha256"]
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128,columns=["text"]):
            for text in batch.column(0).to_pylist():
                normalized=" ".join(unicodedata.normalize("NFKC",text).split())
                h=hashlib.sha256(normalized.encode()).hexdigest()
                if not normalized or h in seen:
                    continue
                seen.add(h)
                if len(prefix)<needed and partition(h)=="train":
                    ids=tokenizer.encode(text,add_special_tokens=False).ids+[manifest["eos"]]
                    if len(ids)>=2:
                        prefix.extend(ids[:needed-len(prefix)])
                if h in samples:
                    split,rows,lengths=samples.pop(h)
                    ids=tokenizer.encode(text,add_special_tokens=False).ids+[manifest["eos"]]
                    for j,(row,n) in enumerate(zip(rows,lengths)):
                        expected=np.asarray(ids[j*512:j*512+int(n)+1],dtype=np.uint16)
                        assert np.array_equal(row[:int(n)+1],expected)
                        assert np.all(row[int(n)+1:]==manifest["eos"])
                    result["sampled_document_checks"].append({"split":split,"document_sha256":h,"windows":len(rows),"source_retokenization_matches":True})
            if len(prefix)==needed and not samples:
                break
        if len(prefix)==needed and not samples:
            break
    assert not samples
    train=np.memmap("runtime/data/train.bin",mode="r",dtype=np.uint16)
    assert np.array_equal(train[:needed],np.asarray(prefix,dtype=np.uint16))
    result["training_prefix_check"]={"tokens":needed,"reconstructed_from_pinned_source":True}
    for seed in [17,42]:
        hashes={}
        for variant in ["vanilla","reset","persistent"]:
            model=StateLoopLM(Config(**frozen["model"],variant=variant),seed=seed)
            state=model.state_dict()
            hashes[variant]={"backbone_and_head":tensor_hash(state,False),"branch":tensor_hash(state,True) if variant!="vanilla" else None}
            del model,state
        assert len({v["backbone_and_head"] for v in hashes.values()})==1
        assert hashes["reset"]["branch"]==hashes["persistent"]["branch"]
        result["pairing"].append({"seed":seed,"hashes":hashes,"passed":True})
    result.update(status="PASSED",cross_split_document_intersection=0,scope="all held-out document hash assignments and disjointness; 12 source documents re-tokenized; first 65536 training tokens reconstructed; full-size initial tensors hashed. Does not prove absence of all near duplicates.")
    dump("experiments/stateloop-20261028-local48/input-audit.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
