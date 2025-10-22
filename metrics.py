"""Metrics module."""


def accuracy(output, target):
    """Compute the accuracy between the `output` and the `target`."""
    _, indices = output.max(dim=1)
    acc = indices.eq(target).float().sum()
    acc = acc / target.size(0)
    return acc
