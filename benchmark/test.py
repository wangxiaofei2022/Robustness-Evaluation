from train import load_model, load_data, test
import argparse
import torch
from utils import load_config, fix_seed, warnings_ignore, updata_config, dataset_from_npy, set_gpu, path_check
import os
from torch.utils.data import Dataset, DataLoader
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='AID')
    parser.add_argument('--attack_model_name', type=str, default='swin_t')
    parser.add_argument('--attack', type=str, default='CW')
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--model_name', type=str, default='resnet50')
    parser.add_argument('--gpu', type=str, default='0,1')
    args = parser.parse_args()
    set_gpu(args.gpu)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = load_config('./config.yaml')
    config = updata_config(args, config)
    warnings_ignore()
    fix_seed(config['seed'])
    save_txt = os.path.join('./result', args.attack + ".txt")


    # attack datasets
    attack_data_path = os.path.join("./dataset/adversarial_data", args.attack_model_name+"_"+args.attack)
    test_dataset = dataset_from_npy(attack_data_path)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, num_workers=2)

    # base models
    net = load_model(config, args, device, pretrain=False, mode='test')

    # test
    acc = test(test_dataloader, net, config, device, attack=args.attack_model_name+"_"+args.attack)

    # with open(save_txt, 'a+') as f:
    #     content = args.attack_model_name + "\t" + args.model_name + "\t" + str(acc) + "\n"
    #     f.write(content)

