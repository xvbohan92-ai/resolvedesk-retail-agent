import mlx.core as mx
import mlx.nn as nn
import numpy as np


def weighted_loss(model, batch, lengths, weights):
    logits = model(batch[:, :-1])
    losses = nn.losses.cross_entropy(logits, batch[:, 1:]).astype(mx.float32)
    # Zero prompt/padding weights; exact tool-name tokens have multiplier 2.
    return (losses * weights).sum() / weights.sum(), (weights > 0).sum()


def iterate_batches(dataset, batch_size, max_seq_length, loop=False, comm_group=None):
    if batch_size != 1 or (comm_group is not None and comm_group.size() != 1):
        raise ValueError("This audited local weighted runner supports one sample and one device only")
    order = sorted(range(len(dataset)), key=lambda i: len(dataset[i]["tokens"]))
    while True:
        indices = np.random.permutation(order) if loop else order
        for index in indices:
            row = dataset[index]
            size = len(row["tokens"])
            if size > max_seq_length:
                raise ValueError("No silent truncation in weighted training")
            padded = min(1 + 32 * ((size + 31) // 32), max_seq_length)
            tokens = row["tokens"] + [0] * (padded - size)
            weights = row["weights"] + [0.0] * (padded - size)
            yield (
                mx.array([tokens]),
                mx.array([[row["prompt_length"], size]]),
                mx.array([weights], dtype=mx.float32),
            )
        if not loop:
            break
