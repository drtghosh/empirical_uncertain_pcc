import torch


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
