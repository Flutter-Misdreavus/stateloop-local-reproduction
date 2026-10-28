import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import torch


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    temp.replace(path)


def commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def clean_code():
    paths = ["stateloop", "scripts", "configs", "tests", "requirements.txt"]
    if subprocess.check_output(["git", "status", "--porcelain", "--", *paths], text=True).strip():
        raise RuntimeError("Execution code/configuration must be committed and clean")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def deadline_epoch():
    from datetime import datetime
    p = json.load(open("configs/protocol.json"))
    return datetime.fromisoformat(p["deadline_utc"].replace("Z", "+00:00")).timestamp()


class TrainData:
    def __init__(self, path, seq_len, seed):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.order = np.random.default_rng(seed + 4242).permutation((len(self.tokens)-1)//seq_len)

    def batch(self, offset, count, device):
        if offset + count > len(self.order):
            raise RuntimeError("Training would repeat data beyond the registered single pass")
        rows = np.stack([self.tokens[i*self.seq_len:i*self.seq_len+self.seq_len+1] for i in self.order[offset:offset+count]]).astype(np.int64)
        t = torch.from_numpy(rows).to(device)
        return t[:, :-1], t[:, 1:]


@torch.no_grad()
def evaluate(model, path, device, micro_batch=4, max_tokens=None, intervention=None, amp=True, include_documents=False):
    was_training = model.training
    model.eval()
    data = np.load(path)
    rows, lengths, docids = data["tokens"], data["lengths"], data["doc_ids"]
    total, ntokens, documents = 0.0, 0, {}
    for i in range(0, len(rows), micro_batch):
        if time.time() >= deadline_epoch():
            raise TimeoutError("Absolute experiment deadline")
        t = torch.from_numpy(rows[i:i+micro_batch].astype(np.int64)).to(device)
        lens = lengths[i:i+micro_batch].astype(np.int64).copy()
        if max_tokens is not None:
            remaining = max_tokens - ntokens
            for j in range(len(lens)):
                lens[j] = min(lens[j], max(0, remaining))
                remaining -= lens[j]
        labels = t[:, 1:].clone()
        mask = torch.arange(labels.shape[1], device=device)[None] < torch.from_numpy(lens).to(device)[:, None]
        labels[~mask] = -100
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            loss = model(t[:, :-1], labels, intervention=intervention, return_token_loss=True)
        sums = loss.double().sum(-1).cpu().numpy()
        for j, (v, n) in enumerate(zip(sums, lens)):
            if n == 0:
                continue
            total += float(v)
            ntokens += int(n)
            if include_documents:
                key = str(docids[i+j])
                entry = documents.setdefault(key, [0.0, 0])
                entry[0] += float(v)
                entry[1] += int(n)
        if max_tokens is not None and ntokens >= max_tokens:
            break
    model.train(was_training)
    if not ntokens or not np.isfinite(total):
        raise RuntimeError("Invalid evaluation result")
    result = {"nll": total/ntokens, "ppl": float(np.exp(total/ntokens)), "nll_sum": total, "tokens": ntokens}
    if include_documents:
        result["documents"] = documents
    return result
