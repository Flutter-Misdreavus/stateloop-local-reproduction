import json
import numpy as np
import torch
from stateloop.model import Config, StateLoopLM
from stateloop.runtime import TrainData, evaluate
from stateloop.assess import bootstrap_difference
from stateloop.train import lr_at


def test_paired_data_order_and_no_repeated_windows(tmp_path):
    path = tmp_path/"train.bin"
    np.arange(1001, dtype=np.uint16).tofile(path)
    a, b, c = [TrainData(path, 16, seed) for seed in [17, 17, 42]]
    assert np.array_equal(a.order, b.order)
    assert not np.array_equal(a.order, c.order)
    assert len(set(a.order)) == len(a.order)
    x, y = a.batch(0, 4, torch.device("cpu"))
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_gradient_accumulation_matches_full_batch():
    cfg = Config(vocab_size=64, layers=1, width=16, heads=2, ffn=32, state_width=16, loops=2, seq_len=8)
    full, split = StateLoopLM(cfg), StateLoopLM(cfg)
    x, y = torch.randint(0, 64, (4,8)), torch.randint(0,64,(4,8))
    full(x,y).backward()
    for i in range(0,4,2):
        (split(x[i:i+2], y[i:i+2])/2).backward()
    for a,b in zip(full.parameters(),split.parameters()):
        assert torch.allclose(a.grad,b.grad,atol=1e-5,rtol=1e-4)


def test_document_aggregation_and_eval_padding(tmp_path):
    cfg = Config(vocab_size=64, layers=1, width=16, heads=2, ffn=32, state_width=16, loops=2, seq_len=8)
    m = StateLoopLM(cfg)
    rows = np.arange(27,dtype=np.uint16).reshape(3,9)
    lengths = np.asarray([8,3,5])
    ids = np.asarray(["document_a","document_a","document_b"])
    path = tmp_path/"eval.npz"
    np.savez(path,tokens=rows,lengths=lengths,doc_ids=ids)
    result = evaluate(m,path,torch.device("cpu"),micro_batch=2,amp=False,include_documents=True)
    assert result["tokens"]==16
    assert result["documents"]["document_a"][1]==11
    assert result["documents"]["document_b"][1]==5
    assert abs(sum(v[0] for v in result["documents"].values())/16-result["nll"])<1e-12
    truncated = evaluate(m,path,torch.device("cpu"),micro_batch=2,amp=False,max_tokens=9)
    assert truncated["tokens"]==9


def test_bootstrap_is_paired_and_schedule_endpoints():
    left={"nll":2.,"documents":{"a":[20.,10],"b":[40.,20]}}
    right={"nll":1.5,"documents":{"a":[15.,10],"b":[30.,20]}}
    result=bootstrap_difference(left,right)
    assert result["difference_nats"]==.5
    assert result["document_bootstrap_95pct"]==[.5,.5]
    assert abs(lr_at(999,1000)-3e-5)<1e-12
    assert lr_at(0,1000)<lr_at(19,1000)
