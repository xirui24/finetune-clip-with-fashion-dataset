import torch
from transformers import TrainingArguments, EarlyStoppingCallback, Trainer

class FashionCLIPTrainer():
    def __init__(self, model, train_dataset, val_dataset, outdir, lr, wd, patience, batch_size, epoch):
        self.training_args = self.get_training_arguments(outdir, lr, wd, batch_size, epoch, patience)
        self.trainer = self.get_trainer(model, self.training_args, train_dataset, val_dataset, patience)

    def collate_fn(self, data):
        pixel_values = torch.stack([example["pixel_values"] for example in data])
        input_ids = torch.tensor([example["input_ids"].tolist() for example in data],dtype=torch.long)
        attention_mask = torch.tensor([example["attention_mask"].tolist() for example in data])
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_loss": True,
        }

    def get_training_arguments(self, outdir, lr, wd, batch_size, epoch, patience):
        return TrainingArguments(output_dir=outdir,
                            do_train=True,
                            weight_decay=wd,
                            dataloader_num_workers=0,
                            per_device_eval_batch_size=batch_size,
                            per_device_train_batch_size=batch_size,
                            num_train_epochs=epoch,
                            evaluation_strategy = "epoch",
                            logging_strategy="epoch",
                            save_strategy="epoch",
                            save_total_limit=patience+1, 
                            eval_steps=8,
                            warmup_steps=0,
                            learning_rate=lr,
                            fp16=False,
                            load_best_model_at_end=True,
                            disable_tqdm=False
                            )

    def get_trainer(self, model, training_args, train_dataset, val_dataset, patience):
        return Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=train_dataset,
                        eval_dataset=val_dataset,
                        data_collator=self.collate_fn,
                        callbacks = [EarlyStoppingCallback(early_stopping_patience = patience)] 
                    )