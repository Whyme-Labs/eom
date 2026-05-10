"""5-minute smoke test: can Unsloth load Gemma-4-E4B + apply LoRA on Modal?

If yes, we know the dep stack is healthy and full SFT can proceed.
If no, the failure mode tells us what's broken without burning a full
training run on the same problem.

Usage:
    modal run scripts/modal_smoke_unsloth.py
"""

from __future__ import annotations

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    # Unsloth's docs recommend the simple `pip install unsloth` and let it
    # resolve transformers/peft/trl/etc. The earlier failures came from us
    # over-pinning transformers and breaking unsloth's own dep contract.
    .pip_install("unsloth")
    .pip_install("bitsandbytes")
)

app = modal.App("eom-smoke", image=image)


@app.function(gpu="A100-40GB", timeout=10 * 60)
def smoke():
    import torch

    print(f"torch={torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        print(f"GPU={torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}")
        print(f"VRAM={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("\nimporting unsloth...")
    from unsloth import FastLanguageModel
    print("unsloth imported OK")

    print("\nloading unsloth/gemma-4-E4B-it-unsloth-bnb-4bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gemma-4-E4B-it-unsloth-bnb-4bit",
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )
    print("model loaded OK")
    print(f"  type={type(model).__name__}")
    print(f"  param count={sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    print("\napplying LoRA...")
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
    print("LoRA applied OK")
    model.print_trainable_parameters()

    print("\ngenerating one sample (tokenizer is multimodal processor — use kwarg)...")
    FastLanguageModel.for_inference(model)
    messages = [{"role": "user", "content": "Reply with the single word: hello"}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inp = tokenizer(text=text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=20, do_sample=False, temperature=0.0)
    print("Output:", tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True))

    print("\nVRAM peak:", torch.cuda.max_memory_reserved() / 1e9, "GB")
    print("\n=== ALL CHECKS PASSED ===")


@app.local_entrypoint()
def main():
    smoke.remote()
