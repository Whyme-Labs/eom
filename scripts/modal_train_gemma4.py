"""Gemma-4-E4B SFT via Unsloth on Modal A100-40GB.

The Unsloth dep stack is the *only* path that supports Gemma-4's
custom layers (Gemma4ClippableLinear, audio_tower, vision_tower).
Smoke-tested at modal_smoke_unsloth.py — load + LoRA + generate works.

Training recipe (Stage 2 stack ported to Gemma-4):
- distill_*.jsonl (minimal student prompt — context distillation, Snell 2022)
- 20 epochs, completion-only loss via manual label mask
- LoRA r=16, α=32, AdamW 8-bit, cosine LR

Usage:
    modal run scripts/modal_train_gemma4.py
"""

from __future__ import annotations

import modal

DATA_DIR = "/home/soh/eom/data/train"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    # Unsloth canonical install — let it own its dep contract.
    .pip_install("unsloth")
    .pip_install("bitsandbytes")
    .add_local_dir(DATA_DIR, remote_path="/data")
)

volume = modal.Volume.from_name("eom-sft-out", create_if_missing=True)
app = modal.App("eom-sft-gemma4", image=image)


@app.function(
    gpu="A100-40GB",
    volumes={"/output": volume},
    timeout=2 * 60 * 60,
)
def train_gemma4():
    import json
    import os
    import time

    import torch
    from datasets import load_dataset
    from transformers import (
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )
    from unsloth import FastLanguageModel

    print(f"torch={torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        print(f"GPU={torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}")

    model_name = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
    max_seq_length = 4096

    model, processor = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    # Gemma-4-E4B is multimodal: returned object is a Processor wrapping a
    # text tokenizer. The DataCollator needs a real tokenizer (with .pad).
    tokenizer = getattr(processor, "tokenizer", processor)
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
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

    # Find the response template within the chat-formatted text so we can
    # mask the prompt portion (assistant-only loss). Gemma-4 follows the
    # same `<start_of_turn>model\n` convention as Gemma-3.
    response_template_str = "<start_of_turn>model\n"
    response_template_ids = tokenizer(
        text=response_template_str, add_special_tokens=False
    )["input_ids"]
    print(f"response template ids: {response_template_ids}")

    def tokenize_with_response_mask(ex):
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
        L = len(response_template_ids)
        for i in range(len(ids) - L + 1):
            if ids[i:i + L] == response_template_ids:
                for j in range(i + L):
                    labels[j] = -100
                break
        return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}

    tokenized = raw.map(tokenize_with_response_mask, remove_columns=raw["train"].column_names)

    def fits(ex):
        return len(ex["input_ids"]) <= max_seq_length

    tokenized = tokenized.filter(fits)
    print("after length filter:", {k: len(v) for k, v in tokenized.items()})

    args = TrainingArguments(
        output_dir="/output/eom-sft-out-gemma4",
        num_train_epochs=20,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="adamw_8bit",
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        fp16=True,
        bf16=False,
        seed=42,
        report_to="none",
        gradient_checkpointing=False,    # Unsloth handles this
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
    )
    print(f"steps/epoch ≈ {len(tokenized['train']) // 8}")

    t0 = time.time()
    stats = trainer.train()
    print(f"trained in {(time.time() - t0)/60:.1f} min")
    print(f"peak VRAM = {torch.cuda.max_memory_reserved()/1e9:.2f} GB")
    print(f"final train loss = {stats.training_loss:.4f}")

    # --- Validation generations ---
    FastLanguageModel.for_inference(model)
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
    out_path = "/output/val-generations-gemma4.jsonl"
    with open(out_path, "w") as f:
        for g in gens:
            f.write(json.dumps(g) + "\n")
    print(f"saved {len(gens)} val generations -> {out_path}")

    # --- Save adapter ---
    out = "/output/eom-sft-adapter-gemma4"
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"adapter saved to {out}")
    print("files:", os.listdir(out))


@app.local_entrypoint()
def main():
    train_gemma4.remote()
