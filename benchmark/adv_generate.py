import os
import torchattacks
import torch
import argparse
import numpy as np
from utils import load_config, fix_seed, warnings_ignore, updata_config, path_check, set_gpu, densnet_key_change
from train import load_model, load_data, test
from tqdm import tqdm

def attacks(net, dataloader, adv_way, device):
    orig_images = []
    adv_images = []
    labels = []
    net.to(device)
    if adv_way == 'FGSM_l':
        attacker = torchattacks.FGSM(net, eps=8/255)
    elif adv_way == 'FGSM_m':
        attacker = torchattacks.FGSM(net, eps=2/255)
    elif adv_way == 'FGSM_s':
        attacker = torchattacks.FGSM(net, eps=0.5/255)
    elif adv_way in ['I-FGSM', 'BIM']:
        attacker = torchattacks.BIM(net, eps=8/255, alpha=2/255, steps=8)
    elif adv_way == 'CW':
        attacker = torchattacks.CW(net, c=0.1, kappa=0, steps=20, lr=0.01)
    elif adv_way == 'PGD-10':
        attacker = torchattacks.PGD(net, eps=8/255, alpha=2/255, steps=10, random_start=False)
    elif adv_way == 'PGD-20':
        attacker = torchattacks.PGD(net, eps=8/255, alpha=2/255, steps=20, random_start=False)
    elif adv_way == 'DeepFool':
        attacker = torchattacks.DeepFool(net)
    elif adv_way == 'MIFGSM':
        attacker = torchattacks.MIFGSM(net, eps=8/255, alpha=2/255, steps=8, decay=1.0)
    elif adv_way == 'AA':
        attacker = torchattacks.AutoAttack(net, eps=8/255, n_classes=30)
    else:
        raise NameError("attacker error")

    for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc=args.model_name + '-' + adv_way, ncols=100)):
        inputs, targets = inputs.to(device), targets.to(device)
        adv_batch = attacker(inputs, targets)
        # orig_images.append(inputs.cpu().detach())
        adv_images.append(adv_batch.cpu().detach())
        labels.append(targets.cpu().detach())

    # orig_images = torch.cat(orig_images, dim=0).numpy()
    adv_images = torch.cat(adv_images, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()

    return adv_images, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='densenet121')
    parser.add_argument('--dataset', type=str, default='AID')
    parser.add_argument('--gpu', type=str, default='5')
    parser.add_argument('--attack', type=str, default='FGSM_l')
    args = parser.parse_args()

    set_gpu(args.gpu)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = load_config('./config.yaml')
    config = updata_config(args, config)
    warnings_ignore()
    fix_seed(config['seed'])

    _, testloader = load_data(config[args.dataset]['path'], config[args.model_name]['batchsize'])
    net = load_model(config, args, device, pretrain=False, mode='test')

    test_adv_images, test_labels = attacks(net, testloader, args.attack, device)
    for i in range(len(test_labels)):
        file_name = str(int(test_labels[i]))
        data = test_adv_images[i]
        path = os.path.join("./dataset/adversarial_data", config['model_name'] + "_" + args.attack, file_name)
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(os.path.join(path, str(i) + '.npy'), data)
    # np.savez(os.path.join(path, config['model_name']+"_"+args.attack +'.npz'), adv_images=test_adv_images,
    #          labels=test_labels)


