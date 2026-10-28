"""Pinned public dataset preparation. No source text is exported to Git."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import unicodedata
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
OUT = Path("runtime/data")
EVIDENCE = Path("experiments/stateloop-20261028-local48")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(url, target, expected=None):
    target = Path(target)
    if target.exists() and (expected is None or sha(target) == expected):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    subprocess.run(["curl", "-fL", "--continue-at", "-", "--retry", "2", "--retry-all-errors", "--connect-timeout", "15", "--max-time", "7200", "-o", str(tmp), url], check=True)
    if expected and sha(tmp) != expected:
        raise RuntimeError("SHA256 mismatch: " + target.name)
    tmp.replace(target)


def split_for(doc_hash):
    bucket = int(hashlib.sha256(b"stateloop-20261028:" + doc_hash).hexdigest()[:16], 16) % 1000
    return "train" if bucket < 975 else "dev" if bucket < 985 else "calibration" if bucket < 990 else "test"


def blocks(tokens):
    for i in range(0, len(tokens) - 511, 512):
        yield hashlib.sha256(np.asarray(tokens[i:i+512], dtype=np.uint16).tobytes()).digest()


def main():
    start = time.time()
    if (OUT / "manifest.json").exists():
        raise RuntimeError("Prepared data already exists; inspect manifest instead of overwriting")
    sources = json.load(open("configs/sources.json"))
    fetch(f"https://huggingface.co/api/datasets/{sources['dataset']}/revision/{sources['dataset_revision']}?blobs=true", "runtime/metadata/dataset.json")
    fetch(f"https://huggingface.co/api/models/{sources['tokenizer']}/revision/{sources['tokenizer_revision']}", "runtime/metadata/tokenizer.json")
    data_meta = json.load(open("runtime/metadata/dataset.json"))
    tok_meta = json.load(open("runtime/metadata/tokenizer.json"))
    rev, trev = data_meta["sha"], tok_meta["sha"]
    assert rev == sources["dataset_revision"] and trev == sources["tokenizer_revision"]
    OUT.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    fetch(f"https://huggingface.co/EleutherAI/pythia-70m/resolve/{trev}/tokenizer.json", "runtime/tokenizer/tokenizer.json")
    tokenizer = Tokenizer.from_file("runtime/tokenizer/tokenizer.json")
    vocab = tokenizer.get_vocab_size()
    eos = tokenizer.token_to_id("<|endoftext|>")
    assert vocab < 65536 and eos is not None
    selected = sorted([s for s in data_meta["siblings"] if s["rfilename"].endswith(".parquet")], key=lambda s: hashlib.sha256(("20261028:" + s["rfilename"]).encode()).hexdigest())
    targets = {"train": 110_000_001, "dev": 1_000_000, "calibration": 500_000, "test": 1_000_000}
    counts = dict.fromkeys(targets, 0)
    buffers = {s: [] for s in targets if s != "train"}
    doc_seen, train_blocks = set(), set()
    consumed, docs, duplicates = [], dict.fromkeys(targets, 0), 0
    with (OUT / "train.bin").open("wb") as train:
        for item in selected[:20]:
            if time.time() - start > 4 * 3600:
                raise RuntimeError("Data preparation exceeded four-hour limit")
            name = item["rfilename"]
            path = OUT / "shards" / Path(name).name
            fetch(f"https://huggingface.co/datasets/EleutherAI/the_pile_deduplicated/resolve/{rev}/{name}", path, item["lfs"]["sha256"])
            consumed.append({"file": name, "sha256": item["lfs"]["sha256"], "bytes": path.stat().st_size})
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=256, columns=["text"]):
                texts, hashes, splits = [], [], []
                for text in batch.column(0).to_pylist():
                    normalized = " ".join(unicodedata.normalize("NFKC", text).split())
                    h = hashlib.sha256(normalized.encode()).digest()
                    if not normalized or h in doc_seen:
                        duplicates += 1
                        continue
                    doc_seen.add(h)
                    split = split_for(h)
                    limit = targets[split] * (1.0 if split == "train" else 1.5)
                    if counts[split] >= limit:
                        continue
                    texts.append(text)
                    hashes.append(h)
                    splits.append(split)
                encoded = tokenizer.encode_batch(texts, add_special_tokens=False)
                for enc, h, split in zip(encoded, hashes, splits):
                    tokens = enc.ids + [eos]
                    if len(tokens) < 2:
                        continue
                    if split == "train":
                        tokens = tokens[:max(0, targets[split] - counts[split])]
                        if not tokens:
                            continue
                        train.write(np.asarray(tokens, dtype=np.uint16).tobytes())
                        train_blocks.update(blocks(tokens))
                        counts[split] += len(tokens)
                    else:
                        buffers[split].append((h, tokens))
                        counts[split] += len(tokens) - 1
                    docs[split] += 1
                if all(counts[s] >= targets[s] * (1.0 if s == "train" else 1.5) for s in targets):
                    break
            print(json.dumps({"stage": "tokenize", "shards": len(consumed), "tokens": counts, "elapsed_s": time.time()-start}), flush=True)
            if all(counts[s] >= targets[s] * (1.0 if s == "train" else 1.5) for s in targets):
                break
    if counts["train"] < targets["train"]:
        raise RuntimeError("Not enough pinned training data")
    artifacts = {"train.bin": {"sha256": sha(OUT / "train.bin"), "tokens": counts["train"], "dtype": "uint16"}}
    heldout_blocks = set()
    discarded = {}
    for split, documents in buffers.items():
        rows, lengths, ids, actual, rejected = [], [], [], 0, 0
        for doc_hash, tokens in documents:
            hashes = set(blocks(tokens))
            if hashes & (train_blocks | heldout_blocks):
                rejected += 1
                continue
            heldout_blocks.update(hashes)
            for offset in range(0, len(tokens)-1, 512):
                part = tokens[offset:offset+513]
                valid = min(len(part)-1, targets[split]-actual)
                if valid <= 0:
                    break
                row = np.full(513, eos, dtype=np.uint16)
                row[:valid+1] = part[:valid+1]
                rows.append(row)
                lengths.append(valid)
                ids.append(doc_hash.hex())
                actual += valid
            if actual >= targets[split]:
                break
        if actual < targets[split]:
            raise RuntimeError(f"Insufficient {split} after contamination filtering: {actual}")
        path = OUT / f"{split}.npz"
        np.savez(path, tokens=np.asarray(rows), lengths=np.asarray(lengths, dtype=np.uint16), doc_ids=np.asarray(ids, dtype="U64"))
        artifacts[path.name] = {"sha256": sha(path), "tokens": actual, "windows": len(rows), "documents": len(set(ids))}
        discarded[split] = rejected
    manifest = {"dataset": data_meta["id"], "revision": rev, "tokenizer_repository": "EleutherAI/pythia-70m", "tokenizer_revision": trev, "tokenizer_sha256": sha("runtime/tokenizer/tokenizer.json"), "vocab_size": vocab, "model_vocab_size": int(np.ceil(vocab/64)*64), "eos": eos, "source_files": consumed, "artifacts": artifacts, "documents_selected": docs, "duplicate_documents_skipped": duplicates, "heldout_documents_with_exact_512_token_overlap_removed": discarded, "split_rule": "normalized NFKC whitespace doc hash; salted SHA256 modulo 1000; train<975/dev<985/calibration<990/test otherwise", "train_packing": "append EOS and concatenate documents within train only", "eval_packing": "document-bounded windows, 512 predicted tokens, mask padding, include EOS", "contamination_scope": "normalized exact document and aligned 512-token exact matches; semantic/near-duplicate absence not guaranteed", "elapsed_s": time.time()-start}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (EVIDENCE / "data-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"status": "DATA_READY", "elapsed_s": time.time()-start}), flush=True)


if __name__ == "__main__":
    main()
