# RecipeForge query parser: QLoRA fine-tune of Qwen2.5-3B-Instruct with Unsloth on a Kaggle T4.
# Outputs (in /kaggle/working): predictions_base.jsonl, predictions_qlora.jsonl, metrics.json, lora/ adapter.
import glob
import json
import os
import subprocess
import sys
import time

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "unsloth"], check=True)

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402
from unsloth import FastLanguageModel  # noqa: E402
from unsloth.chat_templates import train_on_responses_only  # noqa: E402

OUT = "/kaggle/working"
DATA = os.path.dirname(glob.glob("/kaggle/input/**/train.jsonl", recursive=True)[0])
SYSTEM = open(os.path.join(DATA, "system_prompt.txt"), encoding="utf-8").read()
read = lambda f: [json.loads(l) for l in open(os.path.join(DATA, f), encoding="utf-8") if l.strip()]
train_rows, val_rows, test_rows = read("train.jsonl"), read("val.jsonl"), read("test.jsonl")
print("rows", len(train_rows), len(val_rows), len(test_rows), "gpu", torch.cuda.get_device_name(0))
metrics = {"gpu": torch.cuda.get_device_name(0)}

model, tok = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-3B-Instruct-bnb-4bit", max_seq_length=1024, load_in_4bit=True, dtype=None)


def predict(queries, bs=16):
    FastLanguageModel.for_inference(model)
    tok.padding_side = "left"
    outs, t0 = [], time.time()
    for i in range(0, len(queries), bs):
        batch = queries[i:i + bs]
        prompts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
                                           tokenize=False, add_generation_prompt=True) for q in batch]
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=192, do_sample=False, pad_token_id=tok.eos_token_id)
        outs += [tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True) for g in gen]
    return outs, (time.time() - t0) / max(1, len(queries))


def dump(name, queries, raws):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        for q, r in zip(queries, raws):
            f.write(json.dumps({"query": q, "raw": r}, ensure_ascii=False) + "\n")


test_q = [r["query"] for r in test_rows]
raws, per_q = predict(test_q)
dump("predictions_base.jsonl", test_q, raws)
metrics["base_seconds_per_query"] = per_q
print("base done", per_q)

model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0, bias="none", random_state=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth")
FastLanguageModel.for_training(model)
tok.padding_side = "right"
to_text = lambda rows: Dataset.from_list([{"text": tok.apply_chat_template(r["messages"], tokenize=False)} for r in rows])
trainer = SFTTrainer(
    model=model, tokenizer=tok, train_dataset=to_text(train_rows), eval_dataset=to_text(val_rows),
    args=SFTConfig(dataset_text_field="text", max_seq_length=1024, per_device_train_batch_size=8,
                   gradient_accumulation_steps=2, num_train_epochs=1, learning_rate=2e-4, lr_scheduler_type="cosine",
                   warmup_steps=20, logging_steps=25, eval_strategy="steps", eval_steps=125,
                   optim="adamw_8bit", fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
                   output_dir=os.path.join(OUT, "ckpt"), save_strategy="no", report_to="none", seed=0))
trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")
t0 = time.time()
stats = trainer.train()
metrics["train_minutes"] = (time.time() - t0) / 60
metrics["train_loss"] = stats.training_loss
metrics["log_history"] = trainer.state.log_history
print("trained", metrics["train_minutes"], stats.training_loss)

raws, per_q = predict(test_q)
dump("predictions_qlora.jsonl", test_q, raws)
metrics["qlora_seconds_per_query"] = per_q
val_q = [r["messages"][1]["content"] for r in val_rows[:200]]
vraws, _ = predict(val_q)
dump("val_predictions_qlora.jsonl", val_q, vraws)

model.save_pretrained(os.path.join(OUT, "lora"))
tok.save_pretrained(os.path.join(OUT, "lora"))
json.dump(metrics, open(os.path.join(OUT, "metrics.json"), "w"), indent=2, default=str)
print("done")
