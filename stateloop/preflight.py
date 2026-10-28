"""Measured GPU throughput and bounded tiny-data fit before formal training."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time
import torch
from .model import Config, StateLoopLM, optimizer_for
from .runtime import clean_code, commit, deadline_epoch, dump, seed_all


def cfg_for(name, variant):
    p = json.load(open("configs/protocol.json"))
    params = p["models"][name].copy()
    params.update(variant=variant, loops=4, seq_len=512, vocab_size=50304)
    return Config(**params)


def update(model, opt, scaler, x, y):
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model(x, y)
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite loss")
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(norm):
        raise RuntimeError("nonfinite unscaled gradient")
    scaler.step(opt)
    scaler.update()
    return float(loss), float(norm)


def benchmark(candidate):
    records = []
    for variant in ["vanilla", "reset", "persistent"]:
        cfg = cfg_for(candidate, variant)
        seed_all(123)
        model = StateLoopLM(cfg, seed=123).cuda()
        opt = optimizer_for(model)
        scaler = torch.cuda.amp.GradScaler(init_scale=128)
        x = torch.randint(1, cfg.vocab_size, (8, 512), device="cuda")
        y = torch.roll(x, -1, 1)
        torch.cuda.reset_peak_memory_stats()
        for _ in range(3):
            update(model, opt, scaler, x, y)
        torch.cuda.synchronize()
        start = time.monotonic()
        for _ in range(10):
            loss, grad = update(model, opt, scaler, x, y)
        torch.cuda.synchronize()
        elapsed = time.monotonic()-start
        item = {"candidate": candidate, "variant": variant, "config": asdict(cfg), "params": sum(p.numel() for p in model.parameters()), "micro_batch": 8, "measured_steps": 10, "tokens_per_second": 10*8*512/elapsed, "elapsed_s": elapsed, "peak_memory_bytes": torch.cuda.max_memory_allocated(), "last_loss": loss, "last_grad_norm": grad}
        records.append(item)
        print(json.dumps(item), flush=True)
        del model, opt, scaler, x, y
        gc.collect()
        torch.cuda.empty_cache()
    return records


def overfit(candidate):
    results = []
    for variant in ["vanilla", "reset", "persistent"]:
        seed_all(321)
        cfg = cfg_for(candidate, variant)
        m = StateLoopLM(cfg, seed=321).cuda()
        opt = optimizer_for(m, lr=1e-3)
        scaler = torch.cuda.amp.GradScaler(init_scale=128)
        seq = torch.randint(1, 256, (4, 65), device="cuda")
        x, y = seq[:, :-1], seq[:, 1:]
        with torch.no_grad():
            fp32 = m(x).float()
            with torch.autocast("cuda", dtype=torch.float16):
                fp16 = m(x).float()
            error = float((fp32-fp16).abs().max())
        if error > .05:
            raise RuntimeError(f"FP16/FP32 logit discrepancy {error}")
        del fp32, fp16
        initial = None
        start, history = time.monotonic(), []
        for step in range(1, 1001):
            if time.time() >= deadline_epoch() or time.monotonic()-start > 1800:
                raise TimeoutError("Tiny-fit deadline")
            loss, grad = update(m, opt, scaler, x, y)
            if initial is None:
                initial = loss
            if step == 1 or step % 25 == 0 or loss < .1:
                item = {"variant": variant, "step": step, "nll": loss, "gradient_norm": grad}
                history.append(item)
                print(json.dumps(item), flush=True)
            if loss < .1:
                break
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            final = float(m(x, y))
        record = {"candidate": candidate, "variant": variant, "initial_nll": initial, "final_nll": final, "steps": step, "elapsed_s": time.monotonic()-start, "fp16_fp32_initial_max_logit_difference": error, "passed": final < .1, "history": history}
        results.append(record)
        if not record["passed"]:
            raise RuntimeError("Tiny fit did not pass for " + variant)
        del m, opt, scaler, x, y, seq
        gc.collect()
        torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["benchmark", "overfit"])
    parser.add_argument("--candidate", choices=["preferred", "fallback"], default="preferred")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    clean_code()
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    result = {"run_commit": commit(), "mode": args.mode, "candidate": args.candidate, "device": torch.cuda.get_device_name(0), "torch": torch.__version__, "started_epoch": time.time(), "status": "RUNNING"}
    dump(args.output, result)
    try:
        result["records"] = benchmark(args.candidate) if args.mode == "benchmark" else overfit(args.candidate)
        result["status"] = "PASSED"
    except Exception as e:
        result.update(status="FAILED", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        result["ended_epoch"] = time.time()
        dump(args.output, result)


if __name__ == "__main__":
    main()
