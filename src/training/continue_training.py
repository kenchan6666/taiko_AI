import argparse
import torch
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ✅ 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--max_tokens", type=int, default=1024)
parser.add_argument("--gradient_accumulation_steps", type=int, default=32)  # ✅ 默认 32 以减少显存压力
args = parser.parse_args()

# ✅ 设备检查
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ 使用设备: {device}")

# ✅ 加载 Tokenizer 和 Model
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)

# ✅ 加载数据集
dataset = load_dataset("json", data_files=args.data_path, split="train")
dataset = dataset.train_test_split(test_size=0.1)  # ✅ 90% 训练 / 10% 评估
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# ✅ Tokenization
def tokenize_function(examples):
    return tokenizer(examples["title"], truncation=True, padding="max_length", max_length=args.max_tokens)

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# ✅ 训练参数（使用 bf16 而不是 fp16）
training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    evaluation_strategy="steps",
    eval_steps=50,  # ✅ 每 50 步评估一次
    learning_rate=args.lr,
    num_train_epochs=args.epochs,
    bf16=True,  # ✅ 这里改为 bf16（H20 兼容）
    gradient_accumulation_steps=args.gradient_accumulation_steps,  # ✅ 累积梯度，减少显存占用
    save_total_limit=3,
    save_steps=500,
    logging_dir=f"{args.output_dir}/logs",
    logging_steps=50,
    report_to="none",
)

# ✅ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# ✅ 运行训练
trainer.train()
