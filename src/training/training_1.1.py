from transformers import (
    Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer,
    DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch
import datasets

# ✅ **模型路径**
model_path = "/root/autodl-tmp/models/Llama-2-13b-hf"

# ✅ **加载分词器**
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token  # 解决 padding 问题

# ✅ **4-bit 量化配置**
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# ✅ **加载 4-bit 量化模型**
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map="auto"
)

# ✅ **使用 LoRA 进行微调**
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=8,                        # LoRA rank（可调整）
    lora_alpha=32,              # LoRA scaling
    target_modules=["q_proj", "v_proj"],  # 只调整关键层
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# ✅ **应用 LoRA 适配器**
model = get_peft_model(model, peft_config)

print("✅ LoRA 适配器加载完成，开始数据预处理！")

# ✅ **加载数据集**
dataset = datasets.load_dataset("json", data_files="data/taiko_dataset.jsonl", split="train")

# ✅ **Tokenize 数据**
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["notes"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    tokenized["labels"] = tokenized["input_ids"].copy()

    # **处理 padding 以避免 loss 计算错误**
    for i in range(len(tokenized["labels"])):
        tokenized["labels"][i] = [
            -100 if token == tokenizer.pad_token_id else token
            for token in tokenized["labels"][i]
        ]
    return tokenized

if __name__ == "__main__":
    train_test_split = dataset.train_test_split(test_size=0.05)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    train_dataset = train_dataset.map(tokenize_function, batched=True, num_proc=1)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True, num_proc=1)

    # ✅ **DataCollator**
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # **不使用 Masked Language Modeling**
    )

    # ✅ **训练参数**
    training_args = TrainingArguments(
        output_dir="./model/llama2_13B_taiko",
        per_device_train_batch_size=4,    # 适配 96G 显存
        gradient_accumulation_steps=16,   # 累积梯度更新
        num_train_epochs=5,
        eval_strategy="steps",  # ✅ 旧 `evaluation_strategy` 替换为 `eval_strategy`
        eval_steps=250,
        save_steps=250,
        fp16=True,
        logging_dir="./logs",
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        optim="adamw_bnb_8bit",  # ✅ **使用 8-bit AdamW 优化器**
        gradient_checkpointing=True,  # ✅ **启用梯度检查点，减少显存占用**
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        learning_rate=2e-5
    )

    # ✅ **Trainer**
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator  # ✅ **去掉 `tokenizer=tokenizer`**
    )

    # ✅ **开始训练**
    trainer.train()

    # ✅ **保存训练后的模型**
    trainer.save_model("./model/llama2_13B_taiko")
    tokenizer.save_pretrained("./model/llama2_13B_taiko")

    print("✅ 训练完成，模型已保存到 `model/llama2_13B_taiko`")
