"""One fixed-token paired run, with bounded execution and resumable state."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from .model import Config, StateLoopLM, optimizer_for
from .runtime import TrainData, clean_code, commit, deadline_epoch, digest, dump, evaluate, seed_all


def lr_at(step, steps, peak=3e-4):
    warmup = max(1, int(steps*.02))
    if step < warmup:
        return peak*(step+1)/warmup
    fraction = (step-warmup)/max(1, steps-warmup-1)
    return peak*(.1 + .9*.5*(1+math.cos(math.pi*fraction)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["vanilla", "reset", "persistent"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", default="configs/frozen.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wall-deadline", type=float)
    args = parser.parse_args()
    clean_code()
    frozen = json.load(open(args.config))
    if not frozen["ready"]:
        raise RuntimeError("Formal protocol is not ready")
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    seed_all(args.seed)
    cfg = Config(**{**frozen["model"], "variant": args.variant})
    batch, micro = frozen["batch_sequences"], frozen["micro_batch"]
    assert batch % micro == 0
    tokens_step = batch*cfg.seq_len
    target = frozen["tokens_per_run"]
    assert target % tokens_step == 0
    steps = target//tokens_step
    directory = Path("runtime/runs")/args.run_id
    evidence = Path("experiments")/args.run_id
    directory.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    report_path = evidence/"report.json"
    if report_path.exists() and not args.resume:
        raise RuntimeError("Run directory already has a report; use a new run_id or explicit resume")
    data_manifest = json.load(open("runtime/data/manifest.json"))
    if digest("runtime/data/train.bin") != data_manifest["artifacts"]["train.bin"]["sha256"]:
        raise RuntimeError("Training data hash mismatch")
    model = StateLoopLM(cfg, seed=args.seed).to(device)
    opt, scaler = optimizer_for(model), torch.cuda.amp.GradScaler(init_scale=128)
    data = TrainData("runtime/data/train.bin", cfg.seq_len, args.seed)
    checkpoint = directory/"latest.pt"
    begin = time.time()
    run_deadline = min(deadline_epoch(), args.wall_deadline or deadline_epoch())
    report = {"run_id": args.run_id, "status": "RUNNING", "variant": args.variant, "seed": args.seed, "config": asdict(cfg), "params": sum(p.numel() for p in model.parameters()), "run_commit": commit(), "frozen_config_sha256": digest(args.config), "data_manifest_sha256": digest("runtime/data/manifest.json"), "target_tokens": target, "tokens": 0, "steps": 0, "started_epoch": begin, "failures": [], "reliability": "PENDING", "initial_dev": None, "wall_deadline_epoch": run_deadline}
    start_step = 0
    if args.resume:
        old = torch.load(checkpoint, map_location=device)
        if old["report"]["frozen_config_sha256"] != report["frozen_config_sha256"]:
            raise RuntimeError("Resume configuration changed")
        if old["report"]["config"] != asdict(cfg) or old["report"]["seed"] != args.seed:
            raise RuntimeError("Resume identity changed")
        model.load_state_dict(old["model"])
        opt.load_state_dict(old["optimizer"])
        scaler.load_state_dict(old["scaler"])
        start_step = old["step"]
        report = old["report"]
        report.setdefault("resumes", []).append({"epoch": begin, "commit": commit(), "step": start_step})
        report["status"] = "RUNNING"
        torch.set_rng_state(old["rng_torch"].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in old["rng_cuda"]])
        random.setstate(old["rng_python"])
        np.random.set_state(old["rng_numpy"])
    dump(report_path, report)

    def save(step):
        tmp = directory/"latest.pt.tmp"
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(), "step": step, "report": report, "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all(), "rng_python": random.getstate(), "rng_numpy": np.random.get_state()}, tmp)
        tmp.replace(checkpoint)

    try:
        if report["initial_dev"] is None:
            report["initial_dev"] = evaluate(model, "runtime/data/dev.npz", device, micro_batch=micro)
            dump(report_path, report)
        torch.cuda.reset_peak_memory_stats()
        completed = start_step
        train_start, log_start = time.monotonic(), time.monotonic()
        loss_sum, logged_steps = 0.0, 0
        for step in range(start_step, steps):
            if time.time() >= run_deadline:
                save(step)
                raise TimeoutError("Frozen training budget or absolute deadline reached")
            if step % 100 == 0:
                import shutil
                if shutil.disk_usage(".").free < 5*(1<<30):
                    save(step)
                    raise RuntimeError("Less than 5 GiB free disk space")
            opt.zero_grad(set_to_none=True)
            lr = lr_at(step, steps)
            for group in opt.param_groups:
                group["lr"] = lr
            step_loss = 0.0
            for start in range(0, batch, micro):
                x, y = data.batch(step*batch+start, micro, device)
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model(x, y)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                scaler.scale(loss/(batch//micro)).backward()
                step_loss += float(loss)/(batch//micro)
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("Nonfinite training gradient; no silent skipped step")
            scaler.step(opt)
            scaler.update()
            completed = step+1
            report.update(steps=completed, tokens=completed*tokens_step, last_train_nll=step_loss, last_lr=lr, peak_memory_bytes=torch.cuda.max_memory_allocated())
            loss_sum += step_loss
            logged_steps += 1
            if completed % 50 == 0 or completed == steps:
                elapsed = time.monotonic()-log_start
                metric = {"step": completed, "tokens": completed*tokens_step, "train_nll": loss_sum/logged_steps, "lr": lr, "gradient_norm": float(grad_norm), "tokens_per_second": logged_steps*tokens_step/elapsed, "epoch": time.time()}
                with (evidence/"training.jsonl").open("a") as f:
                    f.write(json.dumps(metric)+"\n")
                print(json.dumps(metric), flush=True)
                report["last_progress_epoch"] = time.time()
                dump(report_path, report)
                loss_sum, logged_steps, log_start = 0.0, 0, time.monotonic()
            if completed % 500 == 0 and completed != steps:
                validation = evaluate(model, "runtime/data/dev.npz", device, micro_batch=micro, max_tokens=65536)
                with (evidence/"validation.jsonl").open("a") as f:
                    f.write(json.dumps({"step": completed, "tokens": completed*tokens_step, **validation})+"\n")
            if completed % 1000 == 0:
                save(completed)
        report["training_elapsed_s"] = time.monotonic()-train_start
        report["final_dev"] = evaluate(model, "runtime/data/dev.npz", device, micro_batch=micro)
        gain = report["initial_dev"]["nll"]-report["final_dev"]["nll"]
        report["reliability"] = "MET" if gain >= .5 and report["tokens"] == target else "BELOW_G"
        report["dev_nll_reduction"] = gain
        final = directory/"final.pt"
        torch.save({"model": model.state_dict(), "config": asdict(cfg), "seed": args.seed, "run_commit": report["run_commit"], "tokens": target}, final)
        report["checkpoint"] = {"path": str(final), "sha256": digest(final), "bytes": final.stat().st_size}
        report["status"] = "SUCCEEDED"
        save(completed)
        if report["reliability"] != "MET":
            raise RuntimeError("Formal reliability gate BELOW_G")
    except Exception as e:
        report["status"] = "FAILED"
        report["failures"].append({"type": type(e).__name__, "message": str(e), "epoch": time.time()})
        raise
    finally:
        report["ended_or_updated_epoch"] = time.time()
        report["elapsed_s_this_invocation"] = time.time()-begin
        dump(report_path, report)


if __name__ == "__main__":
    main()
