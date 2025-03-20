from torch import nn


class Triplet(nn.Module):
    def __init__(self, margin=0.1, reduction='mean', p=2):
        super().__init__()

        self.margin = margin
        self.reduction = reduction
        self.p = p

    def forward(self, anchor, positive_key, negative_key):
        return triplet_loss(anchor, positive_key, negative_key, margin=self.margin, reduction=self.reduction, p=self.p)


def triplet_loss(anchor, positive_key, negative_key, margin=0.1, reduction='mean', p=2):
    # Check input dimensionality.
    if anchor.dim() != 2:
        raise ValueError('<anchor> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_key.dim() != 2:
        raise ValueError('<negative_key> must have 2 dimensions.')

    # Check matching number of samples.
    if len(anchor) != len(positive_key):
        raise ValueError('<anchor> and <positive_key> must must have the same number of samples.')
    if len(anchor) != len(negative_key):
        raise ValueError('<anchor> and <negative_key> must must have the same number of samples.')

    # Embedding vectors should have same number of components.
    if anchor.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <anchor> and <positive_key> should have the same number of components.')
    if anchor.shape[-1] != negative_key.shape[-1]:
        raise ValueError('Vectors of <anchor> and <negative_key> should have the same number of components.')

    loss_fn = nn.TripletMarginLoss(margin=margin, p=p, eps=1e-7, reduction=reduction)
    return loss_fn(anchor, positive_key, negative_key)
