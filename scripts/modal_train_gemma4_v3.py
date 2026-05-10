"""Gemma-4-E4B SFT v3 — mirroring Unsloth's official Gemma4_(E4B)-Text Colab.

Differences from v2 (which mode-collapsed to "eom: true"):
1. `FastModel` (not `FastLanguageModel`) — the multimodal-aware loader.
2. LoRA r=8, alpha=8 (was 16/32) — smaller adapter prevents overfit.
3. TRL SFTTrainer + SFTConfig (was vanilla HF Trainer).
4. dtype=None auto-detects bf16 on A100 (fp16 likely caused gradient
   underflow in v2 → mode collapse).
5. weight_decay=0.001, linear LR schedule (was cosine).
6. seed 3407, batch 1 × grad_accum 4.
7. train_on_responses_only applied AFTER trainer construction.

Reference: https://github.com/unslothai/notebooks/blob/main/nb/Gemma4_(E4B)-Text.ipynb
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

volume = modal.Volume.from_name("eom-sft-out", create_if_missing=True)
app = modal.App("eom-sft-gemma4-v3", image=image)


@app.function(
    gpu="A100-40GB",
    volumes={"/output": volume},
    timeout=2 * 60 * 60,
)
def train_gemma4_v3():
    import json
    import os
    import time

    import torch
    from datasets import Dataset, disable_caching, load_dataset
    from transformers import (
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template

    # Avoid datasets fingerprinting (which pickles closures and chokes on the
    # tokenizer's torch._dynamo ConfigModuleInstance).
    disable_caching()

    print(f"torch={torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        print(f"GPU={torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}")

    # Use FastModel for multimodal-aware loading.
    model, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/gemma-4-E4B-it",
        max_seq_length=4096,
        dtype=None,                # auto-detect (bf16 on A100)
        load_in_4bit=True,
        full_finetuning=False,
    )
    # Multimodal Gemma-4 wraps a text tokenizer in a Processor.
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
    tokenizer = get_chat_template(tokenizer, "gemma-4")

    model = FastModel.get_peft_model(
        model,
        r=8,
        lora_alpha=8,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
    )
    model.print_trainable_parameters()

    raw = load_dataset(
        "json",
        data_files={
            "train": "/data/distill_sft.jsonl",
            "val":   "/data/distill_val.jsonl",
        },
    )
    print({k: len(v) for k, v in raw.items()})

    max_seq_length = 4096

    # Build the response-template token sequence so we can mask prompt tokens
    # to -100 manually. Gemma-4 uses `<|turn>model\n` (not Gemma-3's
    # `<start_of_turn>model\n`).
    response_marker = "<|turn>model\n"
    response_ids = tokenizer(
        text=response_marker, add_special_tokens=False
    )["input_ids"]
    print(f"response marker ids: {response_ids}")

    def tokenize_one(ex):
        messages = [
            {"role": "user", "content": ex["input"]},
            {"role": "assistant", "content": ex["target"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        ids = tokenizer(
            text=text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            add_special_tokens=False,
        )["input_ids"]
        labels = list(ids)
        L = len(response_ids)
        masked = False
        for i in range(len(ids) - L + 1):
            if ids[i:i + L] == response_ids:
                for j in range(i + L):
                    labels[j] = -100
                masked = True
                break
        return {
            "input_ids": ids,
            "labels": labels,
            "attention_mask": [1] * len(ids),
            "_masked": masked,
        }

    tokenized = {}
    for split in raw:
        rows = [tokenize_one(ex) for ex in raw[split]]
        n_masked = sum(r["_masked"] for r in rows)
        n_total = len(rows)
        print(f"{split}: {n_masked}/{n_total} examples got response-mask applied")
        rows = [
            {k: v for k, v in r.items() if k != "_masked"}
            for r in rows
            if len(r["input_ids"]) <= max_seq_length
        ]
        tokenized[split] = Dataset.from_list(rows)
    print({k: len(v) for k, v in tokenized.items()})

    args = TrainingArguments(
        output_dir="/output/eom-sft-out-gemma4-v3",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=5,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        bf16=True,                      # A100 supports it; fixes v2 fp16 underflow
        fp16=False,
        gradient_checkpointing=False,   # Unsloth handles this
        remove_unused_columns=False,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True, return_tensors="pt"
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["val"],
        data_collator=collator,
        processing_class=tokenizer,
    )
    print(f"steps/epoch ≈ {len(tokenized['train']) // 4}")

    t0 = time.time()
    stats = trainer.train()
    print(f"trained in {(time.time() - t0)/60:.1f} min")
    print(f"peak VRAM = {torch.cuda.max_memory_reserved()/1e9:.2f} GB")
    print(f"final train loss = {stats.training_loss:.4f}")

    # --- Validation generations ---
    FastModel.for_inference(model)
    raw_val = raw["val"]
    gens = []
    for ex in raw_val:
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
        gens.append({
            "prompt": prompt,
            "generated": tokenizer.decode(
                out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True
            ),
        })
    out_path = "/output/val-generations-gemma4-v3.jsonl"
    with open(out_path, "w") as f:
        for g in gens:
            f.write(json.dumps(g) + "\n")
    print(f"saved {len(gens)} val generations -> {out_path}")

    out = "/output/eom-sft-adapter-gemma4-v3"
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"adapter saved to {out}")
    print("files:", os.listdir(out))


@app.local_entrypoint()
def main():
    train_gemma4_v3.remote()
