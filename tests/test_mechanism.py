import copy
from dataclasses import replace
import io
import torch
from stateloop.model import Config, StateLoopLM, optimizer_for

torch.set_num_threads(2)


def cfg(**kwargs):
    return replace(Config(vocab_size=128, layers=2, width=32, heads=2, ffn=80, state_width=24, seq_len=16, loops=3), **kwargs)


def test_paired_backbones_and_k1_equivalence():
    models = {v: StateLoopLM(cfg(variant=v, loops=1), seed=17) for v in ["vanilla", "reset", "persistent"]}
    reference = models["vanilla"].state_dict()
    for v in ["reset", "persistent"]:
        for name, p in reference.items():
            assert torch.equal(p, models[v].state_dict()[name]), name
    x = torch.randint(0, 128, (2, 16))
    assert (models["reset"](x)-models["persistent"](x)).abs().max() <= 1e-5


def test_zero_reads_and_causality():
    v, p = StateLoopLM(cfg(variant="vanilla")), StateLoopLM(cfg())
    x = torch.randint(0, 128, (2, 16))
    assert (v(x)-p(x, intervention={"all_off": True})).abs().max() <= 1e-5
    changed = x.clone()
    changed[:, 8:] = torch.randint(0, 128, (2, 8))
    for m in [v, p]:
        assert (m(x)[:, :8]-m(changed)[:, :8]).abs().max() <= 1e-5


def test_gate_initialization_and_cross_loop_gradient():
    m = StateLoopLM(cfg())
    assert torch.count_nonzero(m.state.delta.weight) == 0
    delta = torch.nn.functional.softplus(m.state.delta.bias)
    assert torch.allclose(delta, torch.full_like(delta, .3))
    x = torch.randint(0, 128, (2, 16))
    h, states = m.features(x, return_states=True)
    for s in states:
        s.retain_grad()
    m.head(h).square().mean().backward()
    assert states[0].grad is not None and states[0].grad.abs().sum() > 0
    for name in ["write.weight", "delta.weight", "A_log", "feedback.weight", "terminal.weight"]:
        p = dict(m.state.named_parameters())[name]
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name


def test_save_resume_next_update():
    m = StateLoopLM(cfg())
    opt = optimizer_for(m)
    x, y = torch.randint(0, 128, (2, 16)), torch.randint(0, 128, (2, 16))
    def step(model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        model(x, y).backward()
        optimizer.step()
    step(m, opt)
    buf = io.BytesIO()
    torch.save({"model": m.state_dict(), "optimizer": opt.state_dict()}, buf)
    step(m, opt)
    restored = StateLoopLM(cfg())
    ropt = optimizer_for(restored)
    buf.seek(0)
    ckpt = torch.load(buf, map_location="cpu")
    restored.load_state_dict(ckpt["model"])
    ropt.load_state_dict(ckpt["optimizer"])
    step(restored, ropt)
    for a, b in zip(m.parameters(), restored.parameters()):
        assert (a-b).abs().max() <= 1e-5


def test_intervention_single_and_frozen_last_boundary_agree():
    m = StateLoopLM(cfg())
    x = torch.randint(0, 128, (2, 16))
    stats = {"mean": torch.zeros(3, 24), "std": torch.ones(3, 24)}
    single = m(x, intervention={"retain_boundary": 2, "mode": "single", "stats": stats})
    frozen = m(x, intervention={"retain_boundary": 2, "mode": "frozen", "stats": stats})
    assert torch.equal(single, frozen)


def test_masked_nll_matches_manual_and_no_padding_contribution():
    m = StateLoopLM(cfg())
    x, y = torch.randint(0, 128, (2, 16)), torch.randint(0, 128, (2, 16))
    y[1, 9:] = -100
    logits = m(x)
    valid = y != -100
    manual = -logits.float().log_softmax(-1)[valid].gather(-1, y[valid][:, None]).mean()
    assert torch.allclose(manual, m(x, y), atol=1e-6)
