from datasets.buildingpcc import get_dataloader_buildingpcc
from datasets.pcn import get_dataloader_pcn


def get_dataloader(split, config):
    if config.dataset_name == 'buildingpcc':
        return get_dataloader_buildingpcc(split, config)
    elif config.dataset_name == 'pcn':
        return get_dataloader_pcn(split, config)
    else:
        raise ValueError
