
def accuracy(output, target):
    pred, indices = output.max(dim=1)

    acc = indices.eq(target).sum() / target.size(0)
    return acc

