import torch
from torch import nn
from torch.nn import _reduction as _Reduction
from torch import Tensor
from torch.overrides import handle_torch_function, has_torch_function_variadic


def pairwise_margin_loss(
		positive: Tensor,
		negative: Tensor,
		margin: float = 1.0,
		p: float = 2.0,
		reduction: str = "mean",
) -> Tensor:
	"""
	Compute the pairwise margin loss for input tensors using a standard distance function.
	"""

	if has_torch_function_variadic(positive, negative):
		return handle_torch_function(
			pairwise_margin_loss,
			(positive, negative),
			positive,
			negative,
			margin=margin,
			p=p,
			reduction=reduction,
		)

	# Check validity of reduction mode
	if reduction not in ("mean", "sum", "none"):
		raise ValueError(f"{reduction} is not a valid value for reduction")

	# Check validity of margin
	if margin <= 0:
		raise ValueError(f"margin must be greater than 0, got {margin}")

	# Check dimensions
	p_dim = positive.ndim
	n_dim = negative.ndim
	if not (p_dim == n_dim):
		raise RuntimeError(
			f"The anchor, positive, and negative tensors are expected to have "
			f"the same number of dimensions, but got: positive {p_dim}D, and negative {n_dim}D inputs"
		)

	# compute loss
	dist_pos = torch.square(torch.cdist(positive, positive, p=p))
	dist_neg = torch.square(torch.cdist(negative, negative, p=p))
	dist_cross = torch.square(torch.clamp_min(margin - torch.cdist(positive, negative, p=p), 0))

	loss = 0.5 * (dist_pos + dist_neg + dist_cross)

	# Apply reduction
	if reduction == "sum":
		return torch.sum(loss)
	elif reduction == "mean":
		return torch.mean(loss)
	else:  # reduction == "none"
		return loss


class _Loss(nn.Module):
	reduction: str

	def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
		super().__init__()
		if size_average is not None or reduce is not None:
			self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
		else:
			self.reduction = reduction


class PairwiseMarginLoss(_Loss):
	"""
		Creates a criterion that measures the pairwise loss given input
		tensors :math:`pos`, and :math:`neg` (representing positive, and negative examples,
		respectively), and p to compute the standard l_p distance.

		Args:
			margin (float, optional): A nonnegative margin representing the minimum difference
				between the positive and negative distances required for the loss to be 0. Larger
				margins penalize cases where the negative examples are not distant enough from each
				other. Default: :math:`1`.
			p (int, optional): The norm degree for computing distance. Default: :math:`2`.
			reduction (str, optional): Specifies the (optional) reduction to apply to the output:
				``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
				``'mean'``: the sum of the output will be divided by the number of
				elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``


		Shape:
			- Input: :math:`(N, *)` where :math:`*` represents any number of additional dimensions
			as supported by the distance function.
			- Output: A Tensor of shape :math:`(N)` if :attr:`reduction` is ``'none'``, or a scalar
			otherwise.

		Examples::

		>>> pw_loss = PairwiseMarginLoss(margin=1.0, p=2)
		>>> anchor = torch.randn(100, 128, requires_grad=True)
		>>> positive = torch.randn(100, 128, requires_grad=True)
		>>> negative = torch.randn(100, 128, requires_grad=True)
		>>> output = pw_loss(anchor, positive, negative)
		>>> output.backward()
	"""
	__constants__ = ["margin", "p", "reduction"]
	margin: float
	p: int
	reduction: str

	def __init__(
			self,
			*,
			margin: float = 1.0,
			p: float = 2.0,
			reduction: str = "mean",
	):
		super().__init__(size_average=None, reduce=None, reduction=reduction)
		if margin <= 0:
			raise ValueError(
				f"PairwiseMarginLoss: expected margin to be greater than 0, got {margin} instead"
			)
		self.margin = margin
		self.p = p

	def forward(self, positive: Tensor, negative: Tensor) -> Tensor:
		return pairwise_margin_loss(
			positive,
			negative,
			margin=self.margin,
			p=self.p,
			reduction=self.reduction,
		)


def pairwise_loss(positive_key, negative_key, margin=0.1, reduction='mean', p=2):
	# Check input dimensionality.
	if positive_key.dim() != 2:
		raise ValueError('<positive_key> must have 2 dimensions.')
	if negative_key.dim() != 2:
		raise ValueError('<negative_key> must have 2 dimensions.')

	# Check matching number of samples.
	if len(positive_key) != len(negative_key):
		raise ValueError('<positive_key> and <negative_key> must must have the same number of samples.')

	# Embedding vectors should have same number of components.
	if positive_key.shape[-1] != negative_key.shape[-1]:
		raise ValueError('Vectors of <positive_key> and <negative_key> should have the same number of components.')

	loss_fn = PairwiseMarginLoss(margin=margin, p=p, reduction=reduction)
	return loss_fn(positive_key, negative_key)


class PWLoss(nn.Module):
	def __init__(self, margin=0.1, reduction='mean', p=2):
		super().__init__()

		self.margin = margin
		self.reduction = reduction
		self.p = p

	def forward(self, positive_key, negative_key):
		return pairwise_loss(positive_key, negative_key, margin=self.margin, reduction=self.reduction, p=self.p)
