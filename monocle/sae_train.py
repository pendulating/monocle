"""Stage 2 of the sparse-dictionary plan: train a BatchTopK dictionary on the stage-1 shards.

The shards are node-local on klara (`/scratch/mwf62/sae/<arm>/`), so this
module runs there:

    sbatch --time=03:00:00 monocle/monocle.sub monocle.sae_train \
        --arm it --dict P22 --seed 1 --width 32768 --k 48 --epochs 2

    sbatch --time=00:40:00 monocle/monocle.sub monocle.sae_train \
        --arm it --dict A42 --seed 1 --width 32768 --k 48 --epochs 40

The dictionary (Bussmann, Pearce, Leask and Nanda 2024, BatchTopK):

    z_pre = (x - b_dec) @ W_enc + b_enc
    train:  keep the B*k largest ReLU(z_pre) of the batch, zero the rest
    eval:   z = ReLU(z_pre) where z_pre > threshold; the threshold is an
            EMA of the smallest kept value during training (a JumpReLU)
    x_hat  = z @ W_dec + b_dec, with unit-norm rows of W_dec

The loss is the MSE of x_hat plus an auxiliary MSE on the residual error,
reconstructed by the k_aux dead latents with the largest pre-activation,
used without a ReLU so a dead latent gets a gradient (Gao et al. 2024),
weight 1/32. A latent is dead when it did not fire in
the last `dead_window` steps.

The input is the block residual, centered by the training mean and scaled
so the mean vector norm is sqrt(D). The mean and the scale are saved with
the weights. Center first: the gemma residual holds a few massive
coordinates, and a dictionary on the raw residual spends its budget on them.

Data:

- P22 / P40: `shard*_part*.pt` files with `acts` [N, 256, 3840] bf16, train
  images only. `--val-parts` files (a fixed draw, the same for every seed)
  are the validation set and never train.
- A42 / A46: `<frame>/shard*.pt` files with `acts` [N, 3840] bf16 over all
  images and 8 frames. Rows of train-split images train; a fixed 5 percent
  of the train images (by image_path hash) is the validation set. The
  held-out split is not read here.

Outputs under `--out-root/<arm>/<dict>_w<width>_k<k>_s<seed>/`:

    dict.pt       W_enc (fp16), W_dec (fp32), b_enc, b_dec, mean (pooled) or
                  means [n_cond, D] (--center cond: one row per frame or per
                  patch position), scale, threshold, and the config
    metrics.json  FVU, L0 and the dead fraction on the validation set,
                  under both the topk and the threshold encoders
    train_log.csv one row every `--log-every` steps
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import queue
import random
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
SAMPLE_DEFAULT = REPO / "outputs/_monocle/sae/sample/images.parquet"
D = 3840


# --------------------------------------------------------------------------- model


class BatchTopK(torch.nn.Module):
    def __init__(self, d: int, width: int, k: int, seed: int) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        w = torch.randn(width, d, generator=g) / math.sqrt(d)
        w = w / w.norm(dim=1, keepdim=True)
        self.W_dec = torch.nn.Parameter(w)
        self.W_enc = torch.nn.Parameter(w.t().clone().contiguous())
        self.b_enc = torch.nn.Parameter(torch.zeros(width))
        self.b_dec = torch.nn.Parameter(torch.zeros(d))
        self.register_buffer("threshold", torch.tensor(0.0))
        self.k = k
        self.width = width

    def pre(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def encode_topk(self, pre: torch.Tensor) -> torch.Tensor:
        b = pre.shape[0]
        act = F.relu(pre)
        flat = act.flatten()
        kk = min(b * self.k, flat.numel())
        vals, idx = torch.topk(flat, kk, sorted=False)
        z = torch.zeros_like(flat)
        z[idx] = vals
        return z.view_as(pre), vals

    def encode_thr(self, pre: torch.Tensor) -> torch.Tensor:
        act = F.relu(pre)
        return act * (pre > self.threshold)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    @torch.no_grad()
    def unit_decoder(self) -> None:
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def project_decoder_grad(self) -> None:
        # Remove the gradient component parallel to each decoder row, so the
        # step keeps the rows on the unit sphere (Bricken et al. 2023).
        if self.W_dec.grad is None:
            return
        par = (self.W_dec.grad * self.W_dec.data).sum(dim=1, keepdim=True)
        self.W_dec.grad -= par * self.W_dec.data


# --------------------------------------------------------------------------- data


def train_image_set(sample: Path) -> tuple[set[str], set[str]]:
    """The train-split image paths, split into a train set and a 5 percent validation set."""
    df = pd.read_parquet(sample, columns=["image_path", "split"])
    paths = df.loc[df.split == "train", "image_path"].tolist()
    val = {p for p in paths if int(hashlib.md5(p.encode()).hexdigest()[:8], 16) % 20 == 0}
    return set(paths) - val, val


def patch_parts(root: Path, dname: str, val_parts: int) -> tuple[list[Path], list[Path]]:
    parts = sorted((root / dname).glob("shard*_part*.pt"))
    if not parts:
        raise SystemExit(f"no parts under {root / dname}")
    rng = random.Random(0)  # the same validation parts for every seed
    order = parts[:]
    rng.shuffle(order)
    return order[val_parts:], order[:val_parts]


def load_patch_part(f: Path) -> torch.Tensor:
    o = torch.load(f, map_location="cpu", mmap=True, weights_only=False)
    return o["acts"].reshape(-1, D)


def answer_frames(root: Path, dname: str) -> list[str]:
    return sorted(p.name for p in (root / dname).iterdir() if p.is_dir())


def load_answer(root: Path, dname: str, keep: set[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """(rows, frame index per row); the frame order is `answer_frames`."""
    rows, conds = [], []
    for i, frame in enumerate(answer_frames(root, dname)):
        for f in sorted((root / dname / frame).glob("shard*.pt")):
            o = torch.load(f, map_location="cpu", mmap=True, weights_only=False)
            mask = torch.tensor([p in keep for p in o["image_path"]])
            if mask.any():
                rows.append(o["acts"][mask])
                conds.append(torch.full((int(mask.sum()),), i, dtype=torch.long))
    return torch.cat(rows), torch.cat(conds)


class PatchStream:
    """Yields GPU buffers of `buffer_parts` parts, loaded by a background thread."""

    def __init__(self, parts: list[Path], buffer_parts: int, epochs: int, seed: int, device) -> None:
        self.parts, self.buffer_parts, self.epochs, self.device = parts, buffer_parts, epochs, device
        self.rng = random.Random(seed)
        self.q: queue.Queue = queue.Queue(maxsize=1)
        self.n_buffers = epochs * math.ceil(len(parts) / buffer_parts)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        for _ in range(self.epochs):
            order = self.parts[:]
            self.rng.shuffle(order)
            for i in range(0, len(order), self.buffer_parts):
                chunk = order[i : i + self.buffer_parts]
                self.q.put(torch.cat([load_patch_part(f) for f in chunk]))
        self.q.put(None)

    def __iter__(self):
        while True:
            buf = self.q.get()
            if buf is None:
                return
            yield buf.to(self.device)


# --------------------------------------------------------------------------- training


@torch.no_grad()
def cond_mean_scale(x: torch.Tensor, c: torch.Tensor, n_cond: int, chunk: int = 65536) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-condition means [n_cond, D] and the pooled scale (mean centered
    norm over sqrt(D)), in chunks: the mean source can be a million rows, and
    one float32 copy of it does not fit beside the model."""
    sums = torch.zeros(n_cond, D, device=x.device, dtype=torch.float64)
    counts = torch.zeros(n_cond, device=x.device, dtype=torch.float64)
    cc = c if n_cond > 1 else torch.zeros_like(c)
    for i in range(0, x.shape[0], chunk):
        xb = x[i : i + chunk].double()
        sums.index_add_(0, cc[i : i + chunk], xb)
        counts += torch.bincount(cc[i : i + chunk], minlength=n_cond).double()
    means = (sums / counts.clamp_min(1)[:, None]).float()
    norm_sum = 0.0
    for i in range(0, x.shape[0], chunk):
        norm_sum += (x[i : i + chunk].float() - means[cc[i : i + chunk]]).norm(dim=1).sum().item()
    scale = torch.tensor(norm_sum / x.shape[0] / math.sqrt(D), device=x.device)
    return means, scale


def lr_at(step: int, total: int, base: float, warmup: int) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    decay_start = int(total * 0.8)
    if step < decay_start:
        return base
    return base * max(0.0, (total - step) / max(1, total - decay_start))


@torch.no_grad()
def evaluate(model: BatchTopK, x_val: torch.Tensor, batch: int) -> dict:
    """FVU, L0 and the dead fraction on the validation vectors, both encoders."""
    out = {}
    tot_var = ((x_val - x_val.mean(0)) ** 2).sum().item()
    for name in ("topk", "thr"):
        sse = 0.0
        l0 = 0.0
        fires = torch.zeros(model.width, device=x_val.device)
        for i in range(0, x_val.shape[0], batch):
            xb = x_val[i : i + batch].float()
            pre = model.pre(xb)
            z = model.encode_topk(pre)[0] if name == "topk" else model.encode_thr(pre)
            sse += ((model.decode(z) - xb) ** 2).sum().item()
            l0 += (z > 0).sum().item()
            fires += (z > 0).sum(0)
        out[f"fvu_{name}"] = sse / tot_var
        out[f"l0_{name}"] = l0 / x_val.shape[0]
        out[f"dead_frac_{name}"] = (fires == 0).float().mean().item()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["it", "base"])
    ap.add_argument("--dict", required=True, help="P22, P40, A42 or A46")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--width", type=int, default=32768)
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--k-aux", type=int, default=512)
    ap.add_argument("--aux-coef", type=float, default=1 / 32)
    ap.add_argument("--dead-window", type=int, default=500, help="steps without a fire that make a latent dead")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--buffer-parts", type=int, default=4, help="patch parts per shuffled GPU buffer")
    ap.add_argument("--val-parts", type=int, default=2, help="patch parts held out for validation")
    ap.add_argument("--mean-parts", type=int, default=8, help="patch parts that estimate the mean and scale")
    ap.add_argument("--limit-parts", type=int, default=0, help="smoke: train on this many parts")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--data-root", default="/scratch/mwf62/sae")
    ap.add_argument("--out-root", default=str(REPO / "outputs/_monocle/sae"))
    ap.add_argument("--sample", default=str(SAMPLE_DEFAULT))
    ap.add_argument("--tag", default="", help="suffix on the output dir name, for a recipe variant")
    ap.add_argument("--center", choices=["pooled", "cond"], default="pooled",
                    help="cond: subtract the mean per frame (answer) or per patch position (patches)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device("cuda")
    root = Path(args.data_root) / args.arm
    suffix = f"_{args.tag}" if args.tag else ""
    out = Path(args.out_root) / args.arm / f"{args.dict}_w{args.width}_k{args.k}_s{args.seed}{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    is_patch = args.dict.startswith("P")
    t0 = time.time()

    # ---- data, mean and scale
    frames = None
    if is_patch:
        train_parts, val_parts = patch_parts(root, args.dict, args.val_parts)
        if args.limit_parts:
            train_parts = train_parts[: args.limit_parts]
        x_val = torch.cat([load_patch_part(f) for f in val_parts]).to(dev)
        c_val = torch.arange(x_val.shape[0], device=dev) % 256  # rows are [image, position] in order
        x_mean_src = torch.cat([load_patch_part(f) for f in train_parts[: args.mean_parts]]).to(dev)
        c_mean_src = torch.arange(x_mean_src.shape[0], device=dev) % 256
        n_cond = 256
        n_train = len(train_parts) * 512 * 256
        steps_total = args.epochs * math.ceil(n_train / args.batch)
        print(f"[sae-train] {args.arm}/{args.dict}: {len(train_parts)} train parts, "
              f"{len(val_parts)} val parts, {n_train:,} train vectors", flush=True)
    else:
        keep, val_keep = train_image_set(Path(args.sample))
        frames = answer_frames(root, args.dict)
        x_train, c_train = load_answer(root, args.dict, keep)
        x_train, c_train = x_train.to(dev), c_train.to(dev)
        x_val, c_val = load_answer(root, args.dict, val_keep)
        x_val, c_val = x_val.to(dev), c_val.to(dev)
        x_mean_src, c_mean_src = x_train, c_train
        n_cond = len(frames)
        n_train = x_train.shape[0]
        steps_total = args.epochs * math.ceil(n_train / args.batch)
        print(f"[sae-train] {args.arm}/{args.dict}: {n_train:,} train vectors, "
              f"{x_val.shape[0]:,} val vectors ({n_cond} frames)", flush=True)
    # The mean is one row per condition under --center cond, else one row
    # repeated, so the same code path serves both.
    means, scale = cond_mean_scale(x_mean_src, c_mean_src, n_cond if args.center == "cond" else 1)
    if args.center != "cond":
        means = means.expand(n_cond, D).clone()
    del x_mean_src
    print(f"[sae-train] center {args.center}: mean norm {means.norm(dim=1).mean():.1f}, scale {scale:.2f}, "
          f"{steps_total} steps, {time.time() - t0:.0f} s to load", flush=True)

    def norm(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return (x.float() - means[c]) / scale

    x_val = norm(x_val, c_val)

    # ---- model
    model = BatchTopK(D, args.width, args.k, args.seed).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    since_fire = torch.zeros(args.width, device=dev, dtype=torch.long)
    log_f = open(out / "train_log.csv", "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["step", "lr", "mse", "aux", "fvu_batch", "threshold", "dead_frac", "elapsed_s"])

    def batches():
        if is_patch:
            for buf in PatchStream(train_parts, args.buffer_parts, args.epochs, args.seed, dev):
                perm = torch.randperm(buf.shape[0], device=dev)
                for i in range(0, buf.shape[0], args.batch):
                    idx = perm[i : i + args.batch]
                    yield norm(buf[idx], idx % 256)
                del buf
        else:
            for _ in range(args.epochs):
                perm = torch.randperm(n_train, device=dev)
                for i in range(0, n_train, args.batch):
                    idx = perm[i : i + args.batch]
                    yield norm(x_train[idx], c_train[idx])

    step = 0
    for xb in batches():
        lr = lr_at(step, steps_total, args.lr, min(args.warmup, steps_total // 10))
        for g in opt.param_groups:
            g["lr"] = lr
        pre = model.pre(xb)
        z, kept = model.encode_topk(pre)
        x_hat = model.decode(z)
        mse = ((x_hat - xb) ** 2).sum(1).mean()

        # dead-latent auxiliary loss on the residual error
        dead = since_fire >= args.dead_window
        aux = torch.zeros((), device=dev)
        n_dead = int(dead.sum())
        if n_dead > 0:
            err = (xb - x_hat).detach()
            pre_dead = torch.where(dead, pre, torch.full_like(pre, -1e9))
            kk = min(args.k_aux, n_dead)
            vals, idx = torch.topk(pre_dead, kk, dim=1)
            # No ReLU here: a dead latent has a negative pre-activation, and a
            # ReLU would zero its gradient, so it stays dead (the A42 run at
            # width 32k: 70 percent dead, aux loss equal to the error).
            z_aux = torch.zeros_like(pre).scatter(1, idx, vals)
            err_hat = z_aux @ model.W_dec
            aux = ((err_hat - err) ** 2).sum(1).mean()
        loss = mse + args.aux_coef * aux

        opt.zero_grad(set_to_none=True)
        loss.backward()
        model.project_decoder_grad()
        opt.step()
        model.unit_decoder()

        with torch.no_grad():
            fired = (z > 0).any(0)
            since_fire = torch.where(fired, torch.zeros_like(since_fire), since_fire + 1)
            if kept.numel():
                m = kept.min()
                model.threshold = 0.99 * model.threshold + 0.01 * m if step else m
        if step % args.log_every == 0:
            fvu_b = (mse / ((xb - xb.mean(0)) ** 2).sum(1).mean()).item()
            log.writerow([step, f"{lr:.2e}", f"{mse.item():.4f}", f"{aux.item():.4f}", f"{fvu_b:.4f}",
                          f"{model.threshold.item():.4f}", f"{dead.float().mean().item():.4f}",
                          f"{time.time() - t0:.0f}"])
            log_f.flush()
            if step % (args.log_every * 10) == 0:
                print(f"[sae-train] step {step}/{steps_total} mse {mse.item():.4f} fvu {fvu_b:.3f} "
                      f"dead {dead.float().mean().item():.3f} thr {model.threshold.item():.3f} "
                      f"{time.time() - t0:.0f} s", flush=True)
        step += 1
    log_f.close()

    # ---- evaluation and save
    model.eval()
    metrics = evaluate(model, x_val, args.batch)
    metrics.update({
        "arm": args.arm, "dict": args.dict, "seed": args.seed, "width": args.width, "k": args.k,
        "steps": step, "n_train": n_train, "n_val": int(x_val.shape[0]), "epochs": args.epochs,
        "dead_frac_train_window": (since_fire >= args.dead_window).float().mean().item(),
        "threshold": model.threshold.item(), "scale": scale.item(), "mean_norm": means.norm(dim=1).mean().item(),
        "center": args.center, "n_cond": n_cond,
        "train_minutes": (time.time() - t0) / 60,
    })
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    torch.save({
        "W_enc": model.W_enc.data.half().cpu(), "W_dec": model.W_dec.data.cpu(),
        "b_enc": model.b_enc.data.cpu(), "b_dec": model.b_dec.data.cpu(),
        "mean": means[0].cpu() if args.center == "pooled" else means.cpu(),
        "means": means.cpu(), "center": args.center, "frames": frames,
        "scale": scale.cpu(), "threshold": model.threshold.cpu(),
        "config": vars(args),
    }, out / "dict.pt")
    print(f"[sae-train] done: fvu_thr {metrics['fvu_thr']:.3f} l0_thr {metrics['l0_thr']:.1f} "
          f"dead_thr {metrics['dead_frac_thr']:.3f} fvu_topk {metrics['fvu_topk']:.3f} "
          f"in {metrics['train_minutes']:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
