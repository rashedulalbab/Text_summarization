# 📝 Text Summarization with BART

This project fine-tunes the `facebook/bart-base` model on the [DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum) dataset to generate summaries of dialogues.

---

## 🚀 Features
- Uses Hugging Face Transformers & Datasets
- Pretrained model: `facebook/bart-base`
- Fine-tunes on dialogue → summary task
- Evaluation and inference included

---

## 📦 Installation
```bash
pip install transformers datasets fsspec

🔧 Training:


from transformers import Trainer, TrainingArguments

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=8,
        num_train_epochs=2
    ),
    train_dataset=train_data,
    eval_dataset=test_data
)

trainer.train()

📚 Dataset


knkarthick/dialogsum from Hugging Face


