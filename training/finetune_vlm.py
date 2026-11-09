"""LoRA fine-tuning of Qwen3-VL on BigEarthNet.txt (remote-sensing adaptation of the VLM).

Trains low-rank adapters on the language model (attention + MLP projections);
the vision encoder stays frozen (optionally unfreeze the vision-language merger
with --tune-merger). Loss is computed on the answer tokens only.

Runs on a single 16 GB GPU (T4: fp16; L4/A100: bf16). Typical budget with the
default 24k-sample subset: ~3-4 h on an L4, ~6-8 h on a T4, one epoch.

  python training/finetune_vlm.py --data data/bentxt --out checkpoints/satquery-vlm-lora \
      --base Qwen/Qwen3-VL-2B-Instruct --epochs 1 --lr 2e-4 --rank 16

Inference: SATQUERY_VLM_BACKEND=hf SATQUERY_VLM_ADAPTER=checkpoints/satquery-vlm-lora
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satquery.models.vlm_prompts import chat_messages  # noqa: E402


def load_rows(path: Path, limit: int | None = None, seed: int = 0) -> list[dict]:
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def encode(processor, root: Path, row: dict):
    from PIL import Image
    import torch

    images = [Image.open(root / p).convert("RGB") for p in row["images"]]
    full = processor.apply_chat_template(chat_messages(images, row["prompt"], row["answer"]), tokenize=True,
                                         return_dict=True, return_tensors="pt")
    prompt = processor.apply_chat_template(chat_messages(images, row["prompt"]), tokenize=True,
                                           add_generation_prompt=True, return_dict=True, return_tensors="pt")
    ids = full["input_ids"]
    plen = prompt["input_ids"].shape[1]
    if not torch.equal(ids[0, :plen], prompt["input_ids"][0]):
        raise ValueError("chat template prefix mismatch; cannot mask prompt tokens reliably")
    labels = ids.clone()
    labels[:, :plen] = -100
    full["labels"] = labels
    return full


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="folder with train.jsonl / validation.jsonl / images/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0, help="optimizer steps cap (0 = from epochs)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--val-limit", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=200, help="optimizer steps")
    ap.add_argument("--tune-merger", action="store_true", help="also train the vision-language merger")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor, get_cosine_schedule_with_warmup

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = dev == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    processor = AutoProcessor.from_pretrained(args.base)
    model = AutoModelForImageTextToText.from_pretrained(args.base, dtype=torch.bfloat16 if use_bf16 else torch.float32)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lcfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout, bias="none",
                      target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
                      modules_to_save=["merger"] if args.tune_merger else None, task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    model.to(dev)

    train = load_rows(args.data / "train.jsonl", args.train_limit, args.seed)
    val = load_rows(args.data / "validation.jsonl", args.val_limit, args.seed + 1) if (
        args.data / "validation.jsonl").exists() else []
    steps_per_epoch = math.ceil(len(train) / args.grad_accum)
    total = args.max_steps or max(1, int(steps_per_epoch * args.epochs))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(args.warmup * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda" and not use_bf16))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "training_args.json").write_text(json.dumps({**vars(args), "data": str(args.data),
                                                             "out": str(args.out), "n_train": len(train),
                                                             "total_steps": total}, indent=2))

    def run_eval() -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for row in val:
                try:
                    b = encode(processor, args.data, row).to(dev)
                except ValueError:
                    continue
                with torch.autocast(dev, dtype=amp_dtype, enabled=dev == "cuda"):
                    losses.append(float(model(**b).loss))
        model.train()
        return sum(losses) / max(len(losses), 1)

    model.train()
    step, micro, t0, run_loss, best = 0, 0, time.time(), 0.0, float("inf")
    log = (args.out / "train_log.jsonl").open("a", encoding="utf-8")
    epoch = 0
    while step < total:
        random.Random(args.seed + epoch).shuffle(train)
        for row in train:
            try:
                batch = encode(processor, args.data, row).to(dev)
            except (ValueError, OSError) as e:
                print("skip", row.get("id"), e)
                continue
            with torch.autocast(dev, dtype=amp_dtype, enabled=dev == "cuda"):
                loss = model(**batch).loss / args.grad_accum
            scaler.scale(loss).backward()
            run_loss += float(loss.detach())
            micro += 1
            if micro % args.grad_accum:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            rec = {"step": step, "loss": round(run_loss, 4), "lr": sched.get_last_lr()[0],
                   "elapsed_min": round((time.time() - t0) / 60, 1)}
            run_loss = 0.0
            if step % args.eval_every == 0 or step == total:
                if val:
                    rec["val_loss"] = round(run_eval(), 4)
                    if rec["val_loss"] < best:
                        best = rec["val_loss"]
                        model.save_pretrained(args.out)
                        rec["saved"] = True
                else:
                    model.save_pretrained(args.out)
            print(rec, flush=True)
            log.write(json.dumps(rec) + "\n")
            if step >= total:
                break
        epoch += 1
    if not val:
        model.save_pretrained(args.out)
    print(f"done; adapter in {args.out} (best val loss {best:.4f})")


if __name__ == "__main__":
    main()
