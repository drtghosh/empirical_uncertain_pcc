from torch import nn
from torch.nn import _reduction as _Reduction


class _Loss(nn.Module):
	reduction: str

	def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
		super().__init__()
		if size_average is not None or reduce is not None:
			self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
		else:
			self.reduction = reduction
