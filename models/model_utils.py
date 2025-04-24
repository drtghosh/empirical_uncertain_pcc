import torch
from torch.nn import Parameter


def gen_nearest_latents(dci_db, gen_data, complete_data):
    indices = []
    gen_data_sampled = []

    for s in range(gen_data.shape[0]):
        gen_sample = gen_data[s]
        complete_sample = torch.unsqueeze(complete_data[s], 0)
        # try: (adding data)
        dci_db.add(gen_sample)
        # indices, dists = dci_db.query(query, num_neighbours, num_outer_iterations)
        index, _ = dci_db.query(complete_sample, 1, 5000)
        indices.append(index)
        dci_db.clear()
        gen_data_sampled.append(gen_sample[index[0][0].long()])

    # dci_db.free()
    gen_data = torch.stack(gen_data_sampled)
    torch.cuda.empty_cache()

    return gen_data


def _weight_drop(module, weights, dropout, device):
    """
    Helper for `WeightDrop`.
    """

    for name_w in weights:
        w = getattr(module, name_w)
        del module._parameters[name_w]
        module.register_parameter(name_w + '_raw', Parameter(w))

    original_module_forward = module.forward

    def forward(*args, **kwargs):
        for name_w in weights:
            raw_w = getattr(module, name_w + '_raw')
            w = torch.nn.functional.dropout(raw_w, p=dropout, training=module.training).to(device)
            setattr(module, name_w, w)

        return original_module_forward(*args, **kwargs)

    setattr(module, 'forward', forward)


class WeightDrop(torch.nn.Module):
    """
    The weight-dropped module applies recurrent regularization through a DropConnect mask on the
    hidden-to-hidden recurrent weights.

    **Thank you** to Sales Force for their initial implementation of :class:`WeightDrop`. Here is
    their `License
    <https://github.com/salesforce/awd-lstm-lm/blob/master/LICENSE>`__.

    Args:
        module (:class:`torch.nn.Module`): Containing module.
        weights (:class:`list` of :class:`str`): Names of the module weight parameters to apply a
          dropout too.
        dropout (float): The probability a weight will be dropped.

    Example:

        >>> import torch
        >>>
        >>> torch.manual_seed(123)
        <torch._C.Generator object ...
        >>>
        >>> gru = torch.nn.GRUCell(2, 2)
        >>> weights = ['weight_hh']
        >>> weight_drop_gru = WeightDrop(gru, weights, dropout=0.9)
        >>>
        >>> input_ = torch.randn(3, 2)
        >>> hidden_state = torch.randn(3, 2)
        >>> weight_drop_gru(input_, hidden_state)
        tensor(... grad_fn=<AddBackward0>)
    """

    def __init__(self, module, weights, device, dropout=0.0):
        super(WeightDrop, self).__init__()
        _weight_drop(module, weights, dropout, device)
        self.forward = module.forward
