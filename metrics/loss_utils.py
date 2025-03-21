def loss_accumulation(query_data, positive_data, negative_data, batch_size, loss_fn):
    # Check matching number of samples.
    if len(query_data) != len(positive_data):
        raise ValueError(
            f'<query_data> ({len(query_data)}) and <positive_data> ({len(positive_data)}) must have the same number of '
            f'samples.')
    if len(query_data) != len(negative_data):
        raise ValueError(
            f'<query_data> ({len(query_data)}) and <negative_data> ({len(negative_data)}) must have the same number of '
            f'samples.')

    assert len(query_data) % batch_size == 0, 'data size is not divisible by batch size'

    loss = 0
    num_batches = int(len(query_data) / batch_size)
    for n in range(num_batches):
        loss += loss_fn(query_data[n * batch_size:(n + 1) * batch_size],
                        positive_data[n * batch_size:(n + 1) * batch_size],
                        negative_data[n * batch_size:(n + 1) * batch_size])
    loss /= num_batches
    return loss
