import pandas as pd
print("Start Fine Tuning...")

# --------------------
# 1) 数据集
# --------------------
from datasets import load_dataset

dataset = load_dataset("json", data_files="data.jsonl")

def format_example(example):
    return {"text": f"用户: {example['instruction']}\n系统: {example['response']}"}

dataset = dataset.map(format_example)
train_dataset = dataset["train"]

# --------------------
# 2) 模型与分词器
# --------------------
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 重要：多数因果LM没有 pad_token，用 eos 充当 pad，避免 padding 报错
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,      # 如无 bnb，可先改为 False
    device_map="auto"
)

# 训练阶段最好关掉 KV cache
model.config.use_cache = False
model.config.pad_token_id = tokenizer.pad_token_id

# --------------------
# 3) LoRA 轻量微调
# --------------------
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj","v_proj"],  # 如需更强可加 ["k_proj","o_proj"]
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# --------------------
# 4) Tokenize + Collator 生成 labels
# --------------------
from transformers import DataCollatorForLanguageModeling

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

# remove_columns 防止 Trainer 因额外列报错
tokenized_dataset = train_dataset.map(
    tokenize, batched=True, remove_columns=train_dataset.column_names
)

# 关键：让 collator 自动生成 labels（labels = input_ids 的拷贝）
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # 自回归LM，必须是 False
)

# --------------------
# 5) 训练
# --------------------
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="no",
    bf16=False,  # 视显卡支持情况可改为 True/FP16
    fp16=True,   # 如果显卡支持混合精度，能更省显存
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,    # ✅ 这行让 Trainer 拿到 labels，从而有 loss
)

trainer.train()

# --------------------
# 6) 推理测试
# --------------------
model.eval()
prompt = "用户: 你好，我的快递丢了怎么办？\n系统:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
