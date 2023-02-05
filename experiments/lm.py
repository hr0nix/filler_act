import argparse
import yaml
import dataclasses
import random

import numpy as np
import torch as th

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, LogitsWarper
)
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

        return {
            'input_ids': text_with_fillers,
            'attention_mask': [1] * len(text_with_fillers)
        }

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


class EvaluatePerplexityLogitsWarper(LogitsWarper):
    def __init__(self, tokenized_example, eos_token_id):
        self._tokenized_example = tokenized_example
        self._eos_token_id = eos_token_id
        self._loss_sum = 0.0

    @property
    def avg_loss(self):
        return self._loss_sum / len(self._tokenized_example['input_ids'])

    def __call__(self, input_ids, scores):
        assert input_ids.shape[0] == scores.shape[0] == 1, "Only batch size 1 is supported"

        def _single_like(scores, single_index):
            scores = th.full_like(scores, -float('inf'))
            scores[0, -1, single_index] = 0.0
            return scores

        current_len = input_ids.shape[1]
        if current_len == len(self._tokenized_example['input_ids']):
            return _single_like(scores, self._eos_token_id)
        else:
            expected_token = self._tokenized_example['input_ids'][current_len - 1]
            self._loss_sum += scores[0, current_len - 1, expected_token].item()
            return _single_like(scores, expected_token)


def evaluate_perplexity_rolling(model, dataset, tokenizer, num_fillers, device):
    model.eval()
    model.to(device)

    input_ids = tokenizer('', return_tensors='pt').input_ids
    assert input_ids.shape == (0,)
    loss_sum = 0.0
    for example in dataset:
        logits_warper = EvaluatePerplexityLogitsWarper(example, tokenizer.eos_token_id)
        sample = model.sample(input_ids, logits_warper=logits_warper)
        assert sample == input_ids
        loss_sum += logits_warper.avg_loss

    return loss_sum / len(dataset)


def load_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    new_tokens = [FILLER_TOKEN]
    new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())
    tokenizer.add_tokens(list(new_tokens))
    return tokenizer


def train(args):
    config = load_config(args.config_file)

    tokenizer = load_tokenizer(config.base_model)

    model = AutoModelForCausalLM.from_pretrained(config.base_model)
    model = model.to(args.device)
    model.resize_token_embeddings(len(tokenizer))

    dataset = load_dataset(config.dataset, config.dataset_subset)
    dataset = tokenize(tokenizer, dataset)
    dataset = insert_fillers(dataset, tokenizer, config.filler_to_token_ratio, config.no_fillers_prob)
    dataset = batch_texts(dataset, model.config.n_ctx)

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


def eval(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    tokenizer = load_tokenizer(args.tokenizer)
    dataset = load_dataset(args.dataset, args.dataset_subset, split="test")
    dataset = tokenize(tokenizer, dataset)
    evaluate_perplexity_rolling(model, dataset, tokenizer, args.num_fillers, args.device)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode', required=True)

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--config-file', type=str, required=True)
    train_parser.add_argument('--device', type=str, default='cuda')
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser('eval')
    eval_parser.add_argument('--model-path', type=str, required=True)
    eval_parser.add_argument('--tokenizer', type=str, required=True)
    eval_parser.add_argument('--dataset', type=str, required=True)
    eval_parser.add_argument('--dataset-subset', type=str, required=True)
    eval_parser.add_argument('--num-fillers', type=int, required=False, default=0)
    train_parser.add_argument('--device', type=str, default='cuda')
    eval_parser.set_defaults(func=eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
