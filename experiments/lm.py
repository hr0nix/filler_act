import argparse
import yaml
import dataclasses
import random

import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset


FILLER_TOKEN = '<UM>'


@dataclasses.dataclass
class ExperimentConfig:
    dataset: str
    dataset_subset: str
    base_model: str
    num_epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    eval_steps: int
    lr: float
    weight_decay: float
    warmup_ratio: float
    filler_to_token_ratio: float
    no_fillers_prob: float


def load_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return ExperimentConfig(**config)


def add_filler_tokens(tokenizer, model):
    new_tokens = [FILLER_TOKEN]
    new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())
    tokenizer.add_tokens(list(new_tokens))
    model.resize_token_embeddings(len(tokenizer))


def tokenize(tokenizer, dataset):
    def tokenize_fn(example):
        return tokenizer(example['text'])

    return dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=4,
        remove_columns=['text']
    )


def insert_fillers(dataset, tokenizer, filler_to_token_ratio, no_fillers_prob):
    filler_token_index = tokenizer.encode(FILLER_TOKEN)[0]

    def insert_fillers_fn(example):
        if random.random() < no_fillers_prob:
            return example

        num_fillers = int(len(example['input_ids']) * filler_to_token_ratio)
        if num_fillers == 0:
            return example

        num_tokens_after_insert = len(example['input_ids']) + num_fillers
        # Don't insert at the last position because there is no token after it
        filler_indices = np.random.choice(num_tokens_after_insert - 1, num_fillers, replace=False)
        text_pos = 0
        text_with_fillers = []
        for i in range(num_tokens_after_insert):
            if i in filler_indices:
                text_with_fillers.append(filler_token_index)
            else:
                text_with_fillers.append(example['input_ids'][text_pos])
                text_pos += 1

        return {'input_ids': text_with_fillers}

    return dataset.map(insert_fillers_fn, num_proc=4)


def batch_texts(dataset, chunk_length):
    def batch_texts_fn(example):
        concatenated_example = {
            k: sum(example[k], [])
            for k in example.keys()
        }
        total_length = len(concatenated_example['input_ids'])
        total_length = (total_length // chunk_length) * chunk_length
        result = {
            k: [
                v[i: i + chunk_length]
                for i in range(0, total_length, chunk_length)
            ]
            for k, v in concatenated_example.items()
        }
        result['labels'] = result['input_ids'].copy()
        return result

    return dataset.map(
        batch_texts_fn,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )


def run(config, device):
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    model = AutoModelForCausalLM.from_pretrained(config.base_model)

    model = model.to(device)

    dataset = load_dataset(config.dataset, config.dataset_subset)
    print(dataset["train"][0])
    add_filler_tokens(tokenizer, model)
    print(dataset["train"][0])
    dataset = tokenize(tokenizer, dataset)
    print(dataset["train"][0])
    dataset = insert_fillers(dataset, tokenizer, config.filler_to_token_ratio, config.no_fillers_prob)
    print(dataset["train"][0])
    dataset = batch_texts(dataset, model.config.n_ctx)
    print(dataset["train"][0])

    print(f'Train size: {len(dataset["train"])}')
    print(f'Validation size: {len(dataset["validation"])}')
    for i in range(2):
        print(f'Train example {i}: {tokenizer.decode(dataset["train"][i]["input_ids"])}')

    training_args = TrainingArguments(
        output_dir="./model",
        warmup_ratio=config.warmup_ratio,
        logging_strategy="steps",
        save_strategy="steps",
        evaluation_strategy="steps",
        logging_steps=config.eval_steps,
        eval_steps=config.eval_steps,
        save_steps=config.eval_steps,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.lr,
        weight_decay=config.weight_decay,
        num_train_epochs=config.num_epochs,
        report_to="wandb",
        save_total_limit=1,
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    trainer.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    run(config=load_config(args.config_file), device=args.device)


if __name__ == "__main__":
    main()
