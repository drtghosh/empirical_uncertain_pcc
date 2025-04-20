import torch
from torch import nn
from torch import Tensor

from loss_class import _Loss


def compute_spacetime_dist(x1, x2=None, space_dim=None):
	if x2 is None:
		x2 = x1
	else:
		assert x1.size(0) == x2.size(0), "both inputs should have same batch sizes"
		assert x1.size(-1) == x2.size(-1), "both inputs should have same dimensions"
	if space_dim is None:
		space_dim = x1.size(-1) // 2
	time_dim = x1.size(-1) - space_dim
	x1_space, x1_time = torch.split(x1, [space_dim, time_dim], -1)
	x2_space, x2_time = torch.split(x2, [space_dim, time_dim], -1)
	space_dist = torch.square(torch.cdist(x1_space, x2_space))
	time_dist = torch.square(torch.cdist(x1_time, x2_time))
	spacetime_dist = space_dist - time_dist
	return spacetime_dist


def pairwise_spacetime_loss(
		positive: Tensor,
		negative: Tensor,
		threshold: float = 1.0,
		reduction: str = "mean",
) -> Tensor:
	"""
	Compute the combined loss for input positive and negative tensors using a spacetime distance function.
	"""

	# Check validity of reduction mode
	if reduction not in ("mean", "sum", "none"):
		raise ValueError(f"{reduction} is not a valid value for reduction")

	# Check dimensions
	p_dim = positive.ndim
	n_dim = negative.ndim
	if not (p_dim == n_dim):
		raise RuntimeError(
			f"The anchor, positive, and negative tensors are expected to have "
			f"the same number of dimensions, but got: positive {p_dim}D, and negative {n_dim}D inputs"
		)

	# compute loss
	dist_pos = compute_spacetime_dist(positive)
	pos_log_sigmoid = 0.5 * torch.log(nn.Sigmoid()(dist_pos - threshold))
	dist_cross = compute_spacetime_dist(positive, negative)
	cross_log_sigmoid = torch.log(nn.Sigmoid()(threshold - dist_cross))

	loss = pos_log_sigmoid + cross_log_sigmoid

	# Apply reduction
	if reduction == "sum":
		return torch.sum(loss)
	elif reduction == "mean":
		return torch.mean(loss)
	else:  # reduction == "none"
		return loss


class SpaceTimeLoss(_Loss):
	"""
		Creates a criterion that measures the spacetime loss given input
		tensors :math:`pos`, and :math:`neg` (representing positive, and negative examples, respectively),
		and a learnable threshold which acts as the divider between pairwise spacetime distances.

		Args:
			threshold (float): A nonnegative margin representing the minimum difference
				between the positive and negative distances required for the loss to be 0. Larger
				margins penalize cases where the negative examples are not distant enough from each
				other. Default: :math:`1`.
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

		>>> pw_loss = SpaceTimeLoss(threshold=1.0)
		>>> anchor = torch.randn(100, 128, requires_grad=True)
		>>> positive = torch.randn(100, 128, requires_grad=True)
		>>> negative = torch.randn(100, 128, requires_grad=True)
		>>> output = pw_loss(anchor, positive, negative)
		>>> output.backward()
	"""
	__constants__ = ["threshold", "reduction"]
	threshold: float
	reduction: str

	def __init__(
			self,
			*,
			threshold: float = 1.0,
			reduction: str = "mean",
	):
		super().__init__(size_average=None, reduce=None, reduction=reduction)
		self.threshold = threshold

	def forward(self, positive: Tensor, negative: Tensor) -> Tensor:
		return pairwise_spacetime_loss(
			positive,
			negative,
			threshold=self.threshold,
			reduction=self.reduction,
		)
