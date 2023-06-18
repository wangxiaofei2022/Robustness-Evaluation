import torch.backends.cudnn as cudnn
import numpy as np
import torch
import glob
import os
import warnings
import yaml
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader

def load_config(config_file):
    config = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)
    return config


def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def warnings_ignore():
    warnings.filterwarnings('ignore')


def set_gpu(gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu


def path_check(path):
    if isinstance(path, list):
        for item in path:
            if not os.path.exists(item):
                os.makedirs(item)
    elif isinstance(path, str):
        if not os.path.exists(path):
            os.makedirs(path)


def updata_config(args, config):
    para = args.__dict__
    for k, v in para.items():
        # if k in config.keys():
        config[k] = v
    return config


def densnet_key_change(pretrained_state):
    new_state_dict = OrderedDict()
    for k, v in pretrained_state.items():
        if 'denseblock' in k:
            param = k.split(".")
            k1 = ".".join(param[:-3] + [param[-3] + param[-2]] + [param[-1]])
            new_state_dict[k1] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


class dataset_from_npz(Dataset):
    def __init__(self, path):
        self.path = path
        data = np.load(self.path)
        self.adv_images = data['adv_images']
        self.labels = data['labels']

    def __len__(self):
        return self.adv_images.shape[0]

    def __getitem__(self, item):
        return torch.tensor(self.adv_images[item]), torch.tensor(self.labels[item])


class dataset_from_npy(Dataset):
    def __init__(self, path):
        self.file_list = glob.glob(path + '/*/*.npy')

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        file = self.file_list[item]
        label = file.split('/')[-2]
        image = np.load(file)
        return torch.tensor(image), torch.tensor(int(label))
