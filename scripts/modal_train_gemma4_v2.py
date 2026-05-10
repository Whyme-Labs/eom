"""Gemma-4-E4B SFT — v2 with Unsloth's official Gemma-4 recipe.

Fixes from v1:
1. Set chat template via `get_chat_template(tokenizer, "gemma-4")` — Gemma-4
   uses `<|turn>user\\n` / `<|turn>model\\n`, NOT `<start_of_turn>...`.
2. Use `train_on_responses_only(trainer, ...)` for completion-only loss
   instead of manual masking that never matched.
3. `finetune_vision_layers=False` — our data is text-only, save params.
4. L4 GPU (24 GB) — Unsloth docs say E4B trains in 10 GB; A100-40GB was
   massively overkill (~7× cost).
5. 5 epochs (compromise: docs say 1, our small dataset benefits from more).
6. grad_accum 4 (per docs).

Reference: https://unsloth.ai/docs/models/gemma-4/train.md
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
app = modal.App("eom-sft-gemma4-v2", image=image)


@app.function(
    gpu="A100-40GB",               # L4 OOM'd on 4K seq + LoRA (peak >22GB)
    volumes={"/output": volume},
    timeout=2 * 60 * 60,
)
def train_gemma4_v2():
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
    from unsloth.chat_templates import get_chat_template, train_on_responses_only

    print(f"torch={torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        print(f"GPU={torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}")
        print(f"VRAM={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model_name = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
    max_seq_length = 4096

    model, processor = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    # Get the underlying text tokenizer (multimodal Gemma-4 wraps it).
    tokenizer = getattr(processor, "tokenizer", processor)
    # Apply Unsloth's gemma-4 chat template — installs the right
    # <|turn>user/model markers and special tokens.
    tokenizer = get_chat_template(tokenizer, "gemma-4")
    print(f"chat template set; tokenizer={type(tokenizer).__name__}")

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
        finetune_vision_layers=False,       # text-only data
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

    def to_text(ex):
        messages = [
            {"role": "user", "content": ex["input"]},
            {"role": "assistant", "content": ex["target"]},
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    dataset = raw.map(to_text, remove_columns=raw["train"].column_names)

    def tokenize(ex):
        return tokenizer(
            text=ex["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            add_special_tokens=False,
        )

    tokenized = dataset.map(tokenize, remove_columns=["text"])

    def fits(ex):
        return len(ex["input_ids"]) <= max_seq_length

    tokenized = tokenized.filter(fits)
    print("after length filter:", {k: len(v) for k, v in tokenized.items()})

    args = TrainingArguments(
        output_dir="/output/eom-sft-out-gemma4-v2",
        num_train_epochs=5,                # docs recommend 1; we use 5 for small dataset
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,     # per Unsloth docs
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
        gradient_checkpointing=False,      # Unsloth handles this
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
        processing_class=tokenizer,    # train_on_responses_only needs to read this
    )
    # Apply Unsloth's response-only label mask. This wraps the trainer's
    # data collator to set labels[prompt_tokens] = -100 at batch time,
    # using the chat-template's `<|turn>user\n` / `<|turn>model\n` markers.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
    )
    n_train = len(tokenized["train"])
    print(f"steps/epoch ≈ {n_train // 4} (n_train={n_train}, eff_batch=4)")

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
    out_path = "/output/val-generations-gemma4-v2.jsonl"
    with open(out_path, "w") as f:
        for g in gens:
            f.write(json.dumps(g) + "\n")
    print(f"saved {len(gens)} val generations -> {out_path}")

    # --- Save adapter ---
    out = "/output/eom-sft-adapter-gemma4-v2"
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"adapter saved to {out}")
    print("files:", os.listdir(out))


@app.local_entrypoint()
def main():
    train_gemma4_v2.remote()
