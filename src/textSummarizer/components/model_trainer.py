from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from src.textSummarizer.entity import ModelTrainerConfig
import torch
import os


class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def train(self):
        # Clear CUDA cache
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # Load dataset
        dataset = load_from_disk(self.config.data_path)

        # Load pretrained model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(self.device)

        # Enable gradient checkpointing if using GPU
        if self.device == "cuda":
            model.gradient_checkpointing_enable()

        # Data collator
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=self.config.root_dir,
            max_steps=10,                      # Stop training after 10 steps
            per_device_train_batch_size=4,      # Reduce batch size
            weight_decay=self.config.weight_decay,
            logging_steps=500,                  # Reduced logging frequency
            evaluation_strategy="steps",
            eval_steps=10,                    # Reduced evaluation frequency
            save_steps=10,                    # Save less frequently
            save_total_limit=2,                 # Keep fewer checkpoints
            load_best_model_at_end=True,
            gradient_accumulation_steps=2,      # Reduce gradient accumulation
            report_to="none",
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator
        )

        # Check for existing checkpoint
        checkpoint_dir = None
        if os.path.isdir(self.config.root_dir):
            checkpoints = [ckpt for ckpt in os.listdir(self.config.root_dir) if ckpt.startswith("checkpoint-")]
            if checkpoints:
                last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
                checkpoint_dir = os.path.join(self.config.root_dir, last_checkpoint)

                # Verify checkpoint integrity
                if os.path.isdir(checkpoint_dir) and os.path.exists(os.path.join(checkpoint_dir, "pytorch_model.bin")):
                    print(f"Resuming training from checkpoint: {checkpoint_dir}")
                else:
                    print("Found corrupted checkpoint. Starting from scratch.")
                    checkpoint_dir = None

        # Train with checkpoint if available
        trainer.train(resume_from_checkpoint=checkpoint_dir)

        # Save final model
        trainer.save_model(self.config.root_dir)
        print(f"Final model saved at: {self.config.root_dir}")
