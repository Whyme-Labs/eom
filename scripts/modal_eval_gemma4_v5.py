"""Salvage v5: load checkpoint-780 (epoch 30, last saved before preemption)
and run val generations + persist as the final adapter.

The full v5 training run completed (88.8 min, train loss 0.0908) but Modal
preempted the worker before the val-gen + adapter-save phase ran. Per-epoch
checkpoints were saved to /output/eom-sft-out-gemma4-v5/checkpoint-{754,780}.
This script just runs the post-training tail.
"""

from __future__ import annotations

import modal

DATA_DIR = "/home/soh/eom/data/train"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("unsloth")
    .pip_install("bitsandbytes")
    .add_local_dir(DATA_DIR, remote_path="/data")
)

volume = modal.Volume.from_name("eom-sft-out", create_if_missing=False)
app = modal.App("eom-eval-gemma4-v5", image=image)


@app.function(
    gpu="A100-40GB",
    volumes={"/output": volume},
    timeout=30 * 60,
)
def eval_v5():
    import json
    import os
    import shutil
    import time

    import torch
    from datasets import load_dataset
    from peft import PeftModel
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template

    print(f"torch={torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU={torch.cuda.get_device_name(0)}")

    ckpt = "/output/eom-sft-out-gemma4-v5/checkpoint-780"
    print(f"loading base + adapter from {ckpt}")

    model, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/gemma-4-E4B-it",
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
        full_finetuning=False,
    )
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
    tokenizer = get_chat_template(tokenizer, "gemma-4")

    # Attach the LoRA adapter from the saved checkpoint.
    model = PeftModel.from_pretrained(model, ckpt)
    print("adapter attached OK")

    raw = load_dataset(
        "json",
        data_files={"val": "/data/distill_val.jsonl"},
    )
    print(f"val examples: {len(raw['val'])}")

    FastModel.for_inference(model)
    gens = []
    t0 = time.time()
    for i, ex in enumerate(raw["val"]):
        messages = [{"role": "user", "content": ex["input"]}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inp = tokenizer(text=prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inp, max_new_tokens=2048, do_sample=False, temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_text = tokenizer.decode(
            out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True
        )
        gens.append({"prompt": prompt, "generated": gen_text})
        print(f"  gen {i}: {len(gen_text)} chars  (elapsed {time.time()-t0:.0f}s)")

    out_path = "/output/val-generations-gemma4-v5.jsonl"
    with open(out_path, "w") as f:
        for g in gens:
            f.write(json.dumps(g) + "\n")
    print(f"saved {len(gens)} val generations -> {out_path}")

    # Promote the checkpoint to the canonical adapter location.
    dst = "/output/eom-sft-adapter-gemma4-v5"
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(ckpt, dst)
    # Strip Trainer-internal files; keep only adapter + tokenizer.
    keep = {
        "adapter_config.json", "adapter_model.safetensors",
        "tokenizer.json", "tokenizer_config.json",
        "chat_template.jinja", "special_tokens_map.json",
    }
    for f in os.listdir(dst):
        if f not in keep:
            p = os.path.join(dst, f)
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
    print(f"adapter promoted to {dst}")
    print("files:", os.listdir(dst))


@app.local_entrypoint()
def main():
    eval_v5.remote()
