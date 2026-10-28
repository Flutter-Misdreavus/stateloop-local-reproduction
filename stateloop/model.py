"""Paper equations 1--12; token-axis mixer is identity.

Backbone details not fixed in the paper are explicit local choices: bias-free
projections, standard RoPE (base 10000), pre-RMSNorm blocks and final backbone
RMSNorm, independent input/output embeddings, N(0,.02) linear initialization.
"""
import math
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab_size: int = 50304
    layers: int = 8
    width: int = 384
    heads: int = 6
    ffn: int = 1024
    state_width: int = 384
    loops: int = 4
    seq_len: int = 512
    variant: str = "persistent"


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        z = x.float()
        z = z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-6)
        return (z * self.weight).to(x.dtype)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.heads
        self.head_dim = cfg.width // cfg.heads
        self.norm1, self.norm2 = RMSNorm(cfg.width), RMSNorm(cfg.width)
        self.qkv = nn.Linear(cfg.width, 3 * cfg.width, bias=False)
        self.attn_out = nn.Linear(cfg.width, cfg.width, bias=False)
        self.gate_up = nn.Linear(cfg.width, 2 * cfg.ffn, bias=False)
        self.down = nn.Linear(cfg.ffn, cfg.width, bias=False)
        inv = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        angles = torch.outer(torch.arange(cfg.seq_len).float(), inv)
        angles = torch.cat([angles, angles], dim=-1)
        self.register_buffer("cos", angles.cos()[None, None], persistent=False)
        self.register_buffer("sin", angles.sin()[None, None], persistent=False)

    def rope(self, x):
        half = self.head_dim // 2
        rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
        n = x.shape[-2]
        return x * self.cos[:, :, :n].to(x.dtype) + rot * self.sin[:, :, :n].to(x.dtype)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, n, 3, self.heads, self.head_dim).unbind(2)
        q, k, v = [a.transpose(1, 2) for a in (q, k, v)]
        h = F.scaled_dot_product_attention(self.rope(q), self.rope(k), v, dropout_p=0.0, is_causal=True)
        x = x + self.attn_out(h.transpose(1, 2).contiguous().view(b, n, d))
        gate, up = self.gate_up(self.norm2(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(gate) * up)


class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.norm = RMSNorm(cfg.width)

    def forward(self, x):
        for block in self.layers:
            x = block(x)
        return self.norm(x)


class StateBranch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.h_norm, self.s_norm = RMSNorm(cfg.width), RMSNorm(cfg.state_width)
        self.write = nn.Linear(cfg.width, cfg.state_width, bias=False)
        self.delta = nn.Linear(cfg.width, cfg.state_width, bias=True)
        self.A_log = nn.Parameter(torch.linspace(math.log(.01), math.log(4), cfg.state_width))
        self.feedback = nn.Linear(cfg.state_width, cfg.width, bias=False)
        self.terminal = nn.Linear(cfg.state_width, cfg.width, bias=False)

    def reset_gate(self):
        nn.init.zeros_(self.delta.weight)
        nn.init.constant_(self.delta.bias, math.log(math.expm1(.3)))

    def forward(self, h, old, reset=False):
        u = self.h_norm(h)
        p = self.write(u).float()
        delta_logits = self.delta(u).float()
        # No autocast for exponentials or accumulated persistent state.
        with torch.autocast(device_type=h.device.type, enabled=False):
            delta = F.softplus(delta_logits)
            log_g = -self.A_log.float().exp() * delta
            write_rate = -torch.expm1(log_g)
            new = write_rate * p
            if not reset:
                new = new + log_g.exp() * old
        return new


class StateLoopLM(nn.Module):
    def __init__(self, cfg, seed=17):
        super().__init__()
        if cfg.variant not in {"vanilla", "reset", "persistent"}:
            raise ValueError(cfg.variant)
        self.cfg = cfg
        # Branch creation cannot perturb paired backbone/head initialization.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.embedding = nn.Embedding(cfg.vocab_size, cfg.width)
            self.backbone = Backbone(cfg)
            self.head = nn.Linear(cfg.width, cfg.vocab_size, bias=False)
            self.apply(self.initialize)
            if cfg.variant != "vanilla":
                torch.manual_seed(seed + 100000)
                self.state = StateBranch(cfg)
                self.state.apply(self.initialize)
                self.state.reset_gate()

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids, intervention=None, return_states=False):
        intervention = intervention or {}
        h = self.embedding(ids) * math.sqrt(self.cfg.width)
        s = torch.zeros((*ids.shape, self.cfg.state_width), device=ids.device, dtype=torch.float32)
        saved, states = None, []
        for k in range(self.cfg.loops):
            h = self.backbone(h)
            if self.cfg.variant == "vanilla":
                continue
            s = self.state(h, s, reset=self.cfg.variant == "reset")
            boundary = k + 1
            retain = intervention.get("retain_boundary")
            if retain == boundary:
                saved = s
            if retain is not None and boundary > retain:
                replace = intervention["mode"] == "frozen" or boundary == retain + 1
                if replace:
                    stats = intervention["stats"]
                    means, stds = stats["mean"], stats["std"]
                    s = (saved - means[retain-1]) * (stds[boundary-1] / (stds[retain-1] + 1e-6)) + means[boundary-1]
            if return_states:
                states.append(s)
            terminal = k == self.cfg.loops - 1
            masked = intervention.get("all_off", False) or (terminal and intervention.get("terminal_off", False)) or (not terminal and intervention.get("feedback_off", False))
            if not masked:
                read = self.state.s_norm(s)
                h = h + (self.state.terminal(read) if terminal else self.state.feedback(read))
        return (h, states) if return_states else h

    def forward(self, ids, labels=None, intervention=None, return_token_loss=False):
        h = self.features(ids, intervention=intervention)
        logits = self.head(h)
        if labels is None:
            return logits
        losses = F.cross_entropy(logits.float().reshape(-1, self.cfg.vocab_size), labels.reshape(-1), reduction="none", ignore_index=-100).view_as(labels)
        if return_token_loss:
            return losses
        return losses.sum() / (labels != -100).sum()


def optimizer_for(model, lr=3e-4):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 or name.endswith("A_log") else decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": .1}, {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=(.9, .95), eps=1e-8)
