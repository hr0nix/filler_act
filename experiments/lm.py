import argparse
import yaml
import dataclasses
import random
import tqdm

import torch as th

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, LogitsWarper
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
    filler_prob: float


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


def insert_fillers(dataset, tokenizer, filler_prob):
    filler_token_index = tokenizer.convert_tokens_to_ids(FILLER_TOKEN)

    def insert_fillers_fn(example):
        modified_example = []
        example_pos = 0
        while example_pos < len(example['input_ids']):
            if random.random() < filler_prob:
                modified_example.append(filler_token_index)
            else:
                modified_example.append(example['input_ids'][example_pos])
                example_pos += 1

        return {
            'input_ids': modified_example,
            'attention_mask': [1] * len(modified_example),
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


@th.no_grad()
def evaluate_example_loss_rolling(model, tokenizer, tokenized_example, num_fillers):
    filler_token_id = tokenizer.convert_tokens_to_ids(FILLER_TOKEN)

    input_ids = [tokenized_example[0]]
    loss_sum = 0.0
    entropy_sum = 0.0
    cur_fillers = 0
    cur_example_pos = 1
    while cur_example_pos < len(tokenized_example):
        if cur_fillers < num_fillers:
            input_ids.append(filler_token_id)
            cur_fillers += 1
        else:
            input_ids_tensor = th.tensor(input_ids[-model.config.n_ctx:]).to(model.device)
            model_outputs = model(input_ids_tensor)
            scores = model_outputs.logits[-1]

            # Mask out filler probability and renormalize
            scores[filler_token_id] = -float('inf')
            scores = th.log_softmax(scores, dim=-1)

            expected_token = tokenized_example[cur_example_pos]
            loss_sum += -scores[expected_token].item()
            entropy_sum += th.distributions.Categorical(logits=scores).entropy().item()

            input_ids.append(expected_token)
            cur_example_pos += 1
            cur_fillers = 0

    return loss_sum, entropy_sum


@th.no_grad()
def evaluate_loss_rolling(model, dataset, tokenizer, num_fillers, max_example_len, device):
    model = model.eval()
    model = model.to(device)

    loss_sum = 0.0
    entropy_sum = 0.0
    token_count = 0
    for example in tqdm.tqdm(dataset):
        tokenized_example = example['input_ids']

        if len(tokenized_example) == 0:
            # Skip empty examples as the model requires at least one token prompt to generate anything
            continue

        if max_example_len is not None and len(tokenized_example) > max_example_len:
            # Skip examples that are too long if requested
            continue

        loss, entropy = evaluate_example_loss_rolling(model, tokenizer, tokenized_example, num_fillers)
        loss_sum += loss
        entropy_sum += entropy
        # -1 because we don't count the first token, which is used as a prompt
        token_count += len(tokenized_example) - 1

    return loss_sum / token_count, entropy_sum / token_count


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
    dataset = dataset.shuffle()
    # TODO: add BOS/EOS tokens
    dataset = tokenize(tokenizer, dataset)
    # TODO: ideally we should generate new filler positions dynamically every epoch
    dataset = insert_fillers(dataset, tokenizer, config.filler_prob)
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
    loss, entropy = evaluate_loss_rolling(
        model=model, dataset=dataset, tokenizer=tokenizer,
        num_fillers=args.num_fillers, max_example_len=args.max_example_len,
        device=args.device
    )
    print(f'Avg per-token loss: {loss:.3f}')
    print(f'Avg per-token entropy: {entropy:.3f}')


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
    eval_parser.add_argument('--max-example-len', type=int, required=False, default=None)
    eval_parser.add_argument('--device', type=str, default='cuda')
    eval_parser.set_defaults(func=eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
