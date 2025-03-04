from datasets.buildingpcc import get_dataloader_buildingpcc


def get_dataloader(split, config):
    if config.dataset_name == 'buildingpcc':
        return get_dataloader_buildingpcc(split, config)
    else:
        raise ValueError
