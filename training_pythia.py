import gc
import shutil
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)

# Constants
MODEL_NAME = "EleutherAI/pythia-160m"
DATASET_NAME = "fancyzhx/ag_news"

# Devices and clearning
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# Trainer
# Custom loss function to train the autoregressive LLM backbones on the sentence classification task. Given the provided template, the ground-truth prediction token is located at position (-1-int(eos))
def compute_loss_last_token(
    inputs_labels: torch.Tensor,
    outputs_logits: torch.Tensor,
    vocab_size: int,
    reduction: str = "mean",
):
    """Computes the loss that focuses on the token corresponding to the classifier decision"""

    # outputs_logits of size [batch_size, max length sequence in that batch,vocabulary size]

    logits_prediction = outputs_logits[:, -2].contiguous().view(-1, vocab_size)
    labels_to_pred = inputs_labels[:, -1].view(-1)

    loss_ce = torch.nn.CrossEntropyLoss(reduction=reduction)
    return loss_ce(logits_prediction, labels_to_pred)


class CustomTrainer(Trainer):
    def __init__(self, *args, after_c_only_every_n: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.after_c_only_every_n = after_c_only_every_n

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        # Custom loss focused only on the entropy regarding the token predicting the class
        outputs = model(**inputs)
        logits = outputs.logits
        loss = compute_loss_last_token(
            inputs.labels, logits, vocab_size=model.config.vocab_size
        )
        return (loss, outputs) if return_outputs else loss

    def _save_checkpoint(self, model, trial, metrics=None):
        super()._save_checkpoint(model, trial)

        # Deleting the optimizer of the non-last checkpoint
        run_dir = self._get_output_dir(trial=trial)
        current_step = self.state.global_step
        current_folder = f"checkpoint-{current_step}"

        for child in Path(run_dir).iterdir():

            # Check that we are dealing with a checkpoint dir
            # which is not the latest
            if not (
                child.is_dir()
                and "checkpoint-" in child.name
                and child.name != current_folder
            ):
                continue

            # Removing the optimizer and schedler state
            if (child / "optimizer.pt").is_file():
                (child / "optimizer.pt").unlink()

            if (child / "scheduler.pt").is_file():
                (child / "scheduler.pt").unlink()

            # Removing the checkpoint based on self.after_c_only_every_n
            if self.after_c_only_every_n is None:
                continue

            dir_checkpoint_number = int(child.name[len("checkpoint-") :])
            c, n = self.after_c_only_every_n.split(",")
            c, n = int(c), int(n)
            if dir_checkpoint_number > c and not dir_checkpoint_number % n == 0:
                shutil.rmtree(child)


def get_trainer(
    model: PreTrainedModel,
    train_dataset_tokenized: Dataset,
    test_dataset_tokenized: Dataset,
    tokenizer: AutoTokenizer,
) -> Trainer:
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=Path(__file__).parent / "pythia-160m",
        eval_strategy="no",
        learning_rate=2e-5,
        lr_scheduler_type="cosine_with_min_lr",
        warmup_steps=500,
        max_grad_norm=1.0,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        weight_decay=0.01,
        max_steps=3000,
        save_strategy="steps",
        save_steps=50,
        logging_steps=50,
        lr_scheduler_kwargs=dict(min_lr_rate=0),
    )

    print(f"len train dataset : {len(train_dataset_tokenized)}")
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_tokenized,
        eval_dataset=test_dataset_tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer,
        after_c_only_every_n="500,250",
    )

    return trainer


# Templating
def format_template(sample):
    return dict(
        text=f"What type of information is presented in this article?\n\n{sample['text']}\n\nOPTIONS:\n0: World\n1: Sport\n2 :Buisiness\n3: Sci/Tech\nANSWER:{sample['label']}",
        label=sample["label"],
    )


## MAIN
if __name__ == "__main__":
    # Loading
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    dataset = load_dataset(DATASET_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    # Tokenization and templating
    tokenized_ds_train = (
        dataset["train"]
        .map(format_template)
        .map(lambda sample: tokenizer(sample["text"]))
    )
    tokenized_ds_test = (
        dataset["test"]
        .map(format_template)
        .map(lambda sample: tokenizer(sample["text"]))
    )
    tokenized_ds_test = tokenized_ds_test.select(
        np.random.RandomState(42).choice(
            len(tokenized_ds_test), size=1000, replace=False
        )
    )

    # Inner training
    trainer = get_trainer(model, tokenized_ds_train, tokenized_ds_test, tokenizer)
    trainer.train()
