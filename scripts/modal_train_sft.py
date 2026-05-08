"""Modal-hosted SFT training for the EOM project (vanilla transformers+peft).

After 6 attempts with Unsloth on Modal hit cascading TRL/PEFT/transformers
version incompatibilities, switched to plain transformers + peft + Trainer.
Slower than Unsloth but version-stable. fp16 on L4 (sm_89) is fine for
Gemma-3-1B (no need for 4-bit quantization at this size).

Usage (after `modal setup`):
    modal run scripts/modal_train_sft.py

Cost estimate: L4 ~$0.45/hr × ~20 min ≈ $0.15
"""

from __future__ import annotations

import modal

DATA_DIR = "/home/soh/eom/data/train"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        # Stick with the proven vanilla stack from modal-distill (worked end-
        # to-end on Gemma-3-1B). Gemma-4 needs Unsloth, which has its own dep
        # cascade we'll tackle separately.
        "torch==2.5.1",
        "transformers>=4.45,<5",
        "peft>=0.12,<0.15",
        "datasets>=2.20",
        "accelerate>=0.34",
        "sentencepiece",
        "protobuf",
    )
    .add_local_dir(DATA_DIR, remote_path="/data")
)

volume = modal.Volume.from_name("eom-sft-out", create_if_missing=True)
app = modal.App("eom-sft", image=image)


@app.function(
    gpu="A100-40GB",  # E4B in 4-bit is ~3GB; 40GB fits comfortably
    volumes={"/output": volume},
    timeout=2 * 60 * 60,
)
def train_sft():
    import json
    import os
    import time

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    print(f"torch={torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        print(f"GPU={torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}")
        print(f"VRAM={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Stage 2: same Gemma-3-1B base as Stage 1 but with three upgrades to push
    # past 0% pass rate: 20 epochs (vs 5), completion-only loss (manual mask),
    # smaller grad-accum (8 vs 16, more grad steps per epoch).
    # Gemma-4 path is blocked by PEFT not yet supporting Gemma4ClippableLinear.
    model_name = "unsloth/gemma-3-1b-it"
    max_seq_length = 4096

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    raw = load_dataset(
        "json",
        data_files={
            "train": "/data/distill_sft.jsonl",  # student template, no spans menu
            "val":   "/data/distill_val.jsonl",
        },
    )
    print({k: len(v) for k, v in raw.items()})

    # Find the boundary between user prompt and assistant response so we can
    # mask prompt tokens with -100 (assistant-only loss, manual version since
    # Gemma's chat template lacks {% generation %} for TRL's automatic mask).
    response_template_str = "<start_of_turn>model\n"
    response_template_ids = tokenizer(
        response_template_str, add_special_tokens=False
    )["input_ids"]

    def tokenize_with_response_mask(ex):
        messages = [
            {"role": "user", "content": ex["input"]},
            {"role": "assistant", "content": ex["target"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        ids = tokenizer(
            text, truncation=True, max_length=max_seq_length,
            padding=False, add_special_tokens=False,
        )["input_ids"]
        labels = list(ids)
        # Locate the response template within `ids` and mask everything up to
        # and including it.
        L = len(response_template_ids)
        for i in range(len(ids) - L + 1):
            if ids[i:i + L] == response_template_ids:
                for j in range(i + L):
                    labels[j] = -100
                break
        else:
            # Fallback: if marker not found, train on full sequence
            pass
        return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}

    tokenized = raw.map(tokenize_with_response_mask, remove_columns=raw["train"].column_names)

    def fits(ex):
        return len(ex["input_ids"]) <= max_seq_length

    tokenized = tokenized.filter(fits)
    print("after length filter:", {k: len(v) for k, v in tokenized.items()})

    args = TrainingArguments(
        output_dir="/output/eom-sft-out",
        num_train_epochs=20,                      # Stage 2: more passes on the small dataset
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,            # smaller for more grad steps per epoch
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="adamw_torch",
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        fp16=True,
        bf16=False,
        seed=42,
        report_to="none",
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    # Seq2Seq collator pads input_ids with pad_token_id and labels with -100,
    # which is what we need for completion-only loss (already masked prompts).
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

    model.eval()
    gens = []
    raw_val = raw["val"]
    for ex in raw_val:
        messages = [{"role": "user", "content": ex["input"]}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inp = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inp, max_new_tokens=2048, do_sample=False, temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        gens.append({
            "prompt": prompt,
            "generated": tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True),
        })
    with open("/output/val-generations.jsonl", "w") as f:
        for g in gens:
            f.write(json.dumps(g) + "\n")
    print(f"saved {len(gens)} val generations")

    out = "/output/eom-sft-adapter"
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"adapter saved to {out}")
    print("files:", os.listdir(out))


@app.local_entrypoint()
def main():
    train_sft.remote()
