import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
import einops
import argparse
import dataclasses
import yaml
import wandb
import pandas as pd
from functools import reduce


class DataManager:
    PADDING_TOKEN = '<PAD>'
    UNKNOWN_TOKEN = '<UNK>'
    FILLER_TOKEN = '<UM>'

    def __init__(self, example_generator):
        self._example_generator = example_generator
        self._token_to_index = {}
        self._index_to_token = []
        self._max_len = 0

    def replace_generator(self, example_generator):
        result = DataManager(example_generator)
        result._token_to_index = self._token_to_index
        result._index_to_token = self._index_to_token
        result._max_len = self._max_len
        return result

    @property
    def num_tokens(self):
        return len(self._token_to_index)

    @property
    def max_len(self):
        return self._max_len

    def warmup(self, num_seqs=1000):
        # Generate some sequences to figure out the dictionary and max_len
        self.generate_input_target_batch(num_seqs)

    def generate_example(self):
        context, result = self._example_generator()
        self._max_len = max(self._max_len, len(context))
        self._max_len = max(self._max_len, len(result))
        return context, result

    def tokenize_token(self, token):
        token_index = self._token_to_index.get(token)
        if token_index is None:
            token_index = self.num_tokens
            self._index_to_token.append(token)
            self._token_to_index[token] = token_index
        return token_index

    def tokenize(self, seq):
        return [
            self.tokenize_token(token)
            for token in seq
        ]

    def untokenize(self, tokenized_seq):
        seq = []
        for token_index in tokenized_seq:
            seq.append(self._index_to_token[token_index])
        return seq

    def generate_input_target(self):
        context, result = self.generate_example()
        result_input = [self.UNKNOWN_TOKEN] + result[:-1]
        return (
            self.tokenize(context),
            self.tokenize(result_input),
            self.tokenize(result),
        )

    def _pad(self, sequences):
        # Padding would never change max_len, so we don't update it
        max_len = max(len(seq) for seq in sequences)
        padding_token = self.tokenize_token(self.PADDING_TOKEN)
        padded_sequences = [
            seq + [padding_token] * (max_len - len(seq))
            for seq in sequences
        ]
        padding_mask = [
            [True] * len(seq) + [False] * (max_len - len(seq))
            for seq in sequences
        ]
        return padded_sequences, padding_mask

    def generate_input_target_batch(self, batch_size):
        inputs_targets = [self.generate_input_target() for _ in range(batch_size)]
        padded_context, context_mask = self._pad([input_target[0] for input_target in inputs_targets])
        padded_result_input, result_input_mask = self._pad([input_target[1] for input_target in inputs_targets])
        padded_result, result_mask = self._pad([input_target[2] for input_target in inputs_targets])
        assert result_input_mask == result_mask  # Should be the same
        return padded_context, padded_result_input, padded_result, context_mask, result_mask


def generate_addition_example(min_terms, max_terms, min_fillers=0, max_fillers=0):
    num_terms = random.randint(min_terms, max_terms)
    context = [random.randrange(10) for _ in range(num_terms)]
    sum = reduce(lambda x, y: (x + y) % 10, context, 0)
    num_fillers = random.randint(min_fillers, max_fillers)
    result = [DataManager.FILLER_TOKEN] * num_fillers + [sum]
    return context, result


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = th.arange(max_len).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = th.zeros(max_len, d_model)
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = x + self.pe[:x.shape[1]]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, max_len, num_tokens, num_layers, hidden_dim, num_heads):
        super().__init__()
        self._embedding = nn.Embedding(num_embeddings=num_tokens, embedding_dim=hidden_dim)
        self._pos_encoding = PositionalEncoding(d_model=hidden_dim, dropout=0.0, max_len=max_len)
        self._transformer = nn.TransformerDecoder(
            num_layers=num_layers,
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                dim_feedforward=hidden_dim * 4,
                nhead=num_heads,
                norm_first=True,
                batch_first=True,
            ),
        )
        self._logit_producer = nn.Linear(hidden_dim, num_tokens)

    def _generate_causal_mask(self, inputs):
        seq_len = inputs.shape[1]
        mask = th.triu(th.ones(seq_len, seq_len, device=inputs.device, dtype=th.bool))
        mask = mask.transpose(0, 1)
        return mask

    def forward(self, context, context_mask, inputs, inputs_mask):
        assert context_mask.dtype == th.bool
        assert inputs_mask.dtype == th.bool

        context_emb = self._pos_encoding(self._embedding(context))
        input_emb = self._pos_encoding(self._embedding(inputs))

        causal_mask = self._generate_causal_mask(inputs)

        emb = self._transformer(
            tgt=input_emb,
            memory=context_emb,
            tgt_key_padding_mask=~inputs_mask,
            memory_key_padding_mask=~context_mask,
            tgt_mask=~causal_mask,
        )
        logits = self._logit_producer(emb)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self._nll_loss = nn.NLLLoss(reduction='none')

    def forward(self, outputs, targets, mask):
        outputs_flat = einops.rearrange(outputs, "b n d -> (b n) d")
        targets_flat = einops.rearrange(targets, "b n -> (b n)")
        mask_flat = einops.rearrange(mask, "b n -> (b n)")
        loss_flat = self._nll_loss(outputs_flat, targets_flat)
        loss = th.mean(loss_flat * mask_flat)
        return loss


def get_model_device(model):
    return next(model.parameters()).device


def train_model(model, data_manager, num_train_batches, num_val_batches, num_epochs, batch_size):
    device = get_model_device(model)
    optimizer = th.optim.Adam(params=model.parameters(), lr=1e-3)

    loss = Loss()

    @th.no_grad()
    def make_batch():
        context, result_input, result, context_mask, result_mask = \
            data_manager.generate_input_target_batch(batch_size)
        context = th.tensor(context, dtype=th.long, device=device)
        result_input = th.tensor(result_input, dtype=th.long, device=device)
        result = th.tensor(result, dtype=th.long, device=device)
        context_mask = th.tensor(context_mask, dtype=th.bool, device=device)
        result_mask = th.tensor(result_mask, dtype=th.bool, device=device)
        return context, result_input, result, context_mask, result_mask

    for epoch in range(num_epochs):
        train_loss_sum = 0.0
        for batch_index in range(num_train_batches):
            optimizer.zero_grad()

            context, result_input, result, context_mask, result_mask = make_batch()
            outputs = model(
                context=context, context_mask=context_mask,
                inputs=result_input, inputs_mask=result_mask,
            )

            loss_value = loss(outputs, result, result_mask)
            train_loss_sum += float(loss_value.item())
            loss_value.backward()
            optimizer.step()

        val_loss_sum = 0.0
        with th.no_grad():
            for batch_index in range(num_val_batches):
                context, result_input, result, context_mask, result_mask = make_batch()
                outputs = model(
                    context=context, context_mask=context_mask,
                    inputs=result_input, inputs_mask=result_mask,
                )
                loss_value = loss(outputs, result, result_mask)
                val_loss_sum += float(loss_value.item())

        wandb.log({
            'train_loss': train_loss_sum / num_train_batches,
            'val_loss': val_loss_sum / num_val_batches,
        }, step=epoch)

    # Save model to wandb
    th.save(model.state_dict(), 'model.pt')
    wandb.save('model.pt', policy='now')

    return model


@th.no_grad()
def compute_accuracy(model, data_manager, num_iters=1000, last_n_tokens=None, initial_input=None, ban_tokens=None):
    device = next(model.parameters()).device
    sum_correct = 0
    for i in range(num_iters):
        context, _, result = data_manager.generate_input_target()
        tokens_to_generate = len(result)
        input = data_manager.tokenize(initial_input or [DataManager.UNKNOWN_TOKEN])
        assert len(input) > 0 and input[0] == data_manager.tokenize_token(DataManager.UNKNOWN_TOKEN)
        for j in range(len(input) - 1, tokens_to_generate):
            # Yep, we do O(n^2) inference here
            context_tensor = th.tensor([context], device=device, dtype=th.long)
            input_tensor = th.tensor([input], device=device, dtype=th.long)
            token_log_probs = model(
                context=context_tensor,
                context_mask=th.ones_like(context_tensor, dtype=th.bool),
                inputs=input_tensor,
                inputs_mask=th.ones_like(input_tensor, dtype=th.bool),
            )

            if ban_tokens is not None:
                for token in ban_tokens:
                    token_index = data_manager.tokenize_token(token)
                    token_log_probs[0, -1, token_index] = -100000
                token_log_probs = F.log_softmax(token_log_probs, dim=-1)

            token_probs = np.exp(token_log_probs[0, -1].cpu().numpy())
            sampled_token = np.random.choice(len(token_probs), p=token_probs)
            input.append(sampled_token)

        if last_n_tokens is None:
            sum_correct += input[1:] == result
        else:
            sum_correct += input[-last_n_tokens:] == result[-last_n_tokens:]

    return float(sum_correct) / num_iters


def evaluate_model(model, data_manager, config):
    dataframe = {
        'terms': [],
        'fillers': [],
        'accuracy': [],
    }
    for num_terms in range(config.min_terms, config.max_terms + 1):
        for num_fillers in range(config.min_fillers, config.max_fillers + 1):
            # Preserve dictionary but change token distribution
            dm = data_manager.replace_generator(
                lambda: generate_addition_example(
                    min_terms=num_terms, max_terms=num_terms,
                    min_fillers=num_fillers, max_fillers=num_fillers,
                )
            )
            initial_input = [DataManager.UNKNOWN_TOKEN] + [DataManager.FILLER_TOKEN] * num_terms
            accuracy = compute_accuracy(
                model, data_manager,
                num_iters=1000, last_n_tokens=1, initial_input=initial_input, ban_tokens=[DataManager.FILLER_TOKEN],
            )

            dataframe['terms'].append(num_terms)
            dataframe['fillers'].append(num_fillers)
            dataframe['accuracy'].append(accuracy)

    wandb.log({'accuracy': pd.DataFrame(dataframe)})


@dataclasses.dataclass
class ExperimentConfig:
    num_train_batches: int
    num_val_batches: int
    num_epochs: int
    batch_size: int
    num_layers: int
    num_heads: int
    hidden_dim: int
    min_terms: int
    max_terms: int
    min_fillers: int
    max_fillers: int


def load_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return ExperimentConfig(**config)


def run(config, device):
    wandb.init(project='filler_act', config=config)

    data_manager = DataManager(
        lambda: generate_addition_example(
            min_terms=config.min_terms, max_terms=config.max_terms,
            min_fillers=config.min_fillers, max_fillers=config.max_fillers,
        )
    )
    data_manager.warmup()  # To figure out the vocabulary and sequence length in advance

    model = TransformerModel(
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        hidden_dim=config.hidden_dim,
        max_len=data_manager.max_len,
        num_tokens=data_manager.num_tokens,
    ).to(device)

    model = train_model(
        model=model,
        data_manager=data_manager,
        num_train_batches=config.num_train_batches,
        num_val_batches=config.num_val_batches,
        num_epochs=config.num_epochs,
        batch_size=config.batch_size,
    )

    evaluate_model(model, data_manager, config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    run(config=load_config(args.config_file), device=args.device)


if __name__ == '__main__':
    main()
