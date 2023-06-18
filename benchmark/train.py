import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import models
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from utils import load_config, fix_seed, warnings_ignore, updata_config, path_check, set_gpu, densnet_key_change
import argparse
from tqdm import tqdm


def load_data(path, batch_size):
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dir = os.path.join(path, "train")
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    train_dataloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=4)

    test_dir = os.path.join(path, "test")
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
    test_dataloader = DataLoader(test_datasets, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_dataloader, test_dataloader



def load_model(config, args, device, pretrain=True, mode='train'):
    if config['model_name'] == 'densenet121':
        net = models.densenet121(pretrained=False)
    elif config['model_name'] == 'densenet169':
        net = models.densenet169(pretrained=False)
    elif config['model_name'] == 'densenet201':
        net = models.densenet201(pretrained=False)
    elif config['model_name'] == 'mobilenet_v2':
        net = models.mobilenet_v2(pretrained=False)
    elif config['model_name'] == 'mobilenet_v3_large':
        net = models.mobilenet_v3_large(pretrained=False)
    elif config['model_name'] == 'mobilenet_v3_small':
        net = models.mobilenet_v3_small(pretrained=False)
    elif config['model_name'] == 'resnet101':
        net = models.resnet101(pretrained=False)
    elif config['model_name'] == 'resnet152':
        net = models.resnet152(pretrained=False)
    elif config['model_name'] == 'resnet18':
        net = models.resnet18(pretrained=False)
    elif config['model_name'] == 'resnet34':
        net = models.resnet34(pretrained=False)
    elif config['model_name'] == 'resnet50':
        net = models.resnet50(pretrained=False)
    elif config['model_name'] == 'resnext101_32x8d':
        net = models.resnext101_32x8d(pretrained=False)
    elif config['model_name'] == 'resnext50_32x4d':
        net = models.resnext50_32x4d(pretrained=False)
    elif config['model_name'] == 'shufflenet_v2_x0_5':
        net = models.shufflenet_v2_x0_5(pretrained=False)
    elif config['model_name'] == 'shufflenet_v2_x1_0':
        net = models.shufflenet_v2_x1_0(pretrained=False)
    elif config['model_name'] == 'shufflenet_v2_x1_5':
        net = models.shufflenet_v2_x1_5(pretrained=False)
    elif config['model_name'] == 'shufflenet_v2_x2_0':
        net = models.shufflenet_v2_x2_0(pretrained=False)
    elif config['model_name'] == 'swin_s':
        net = models.swin_s(weights=None)
    elif config['model_name'] == 'swin_t':
        net = models.swin_t(weights=None)
    elif config['model_name'] == 'vit_b_16':
        net = models.vit_b_16(weights=None)
    elif config['model_name'] == 'vit_b_32':
        net = models.vit_b_32(weights=None)
    elif config['model_name'] == 'wide_resnet101_2':
        net = models.wide_resnet101_2(pretrained=False)
    elif config['model_name'] == 'wide_resnet50_2':
        net = models.wide_resnet50_2(pretrained=False)
    else:
        raise NameError("model error")
    if mode == 'train' and pretrain:
        model_pre = config[config['model_name']]['pretrain']
        ckpt = torch.load(model_pre)
        if config['model_name'][:8] == 'densenet':
            ckpt = densnet_key_change(ckpt)
        net.load_state_dict(ckpt)
    net.fc = nn.Linear(config[args.model_name]['num_linear'], config[config['dataset']]['num_class'])
    if mode == 'test':
        model_trained = os.path.join(config['model_save_path'], config['model_name'] + '.pth')
        ckpt = torch.load(model_trained)
        net.load_state_dict(ckpt['net'])
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    return net


def train(epoch):
    print('Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, desc='{}-{}-{}'.format(config['dataset'],config['model_name'] , epoch), ncols=100)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print('train\tLoss:{:.3f}\taccuracy:{:.3f}%'.format(train_loss / len(trainloader), 100. * correct / total))
    return 100. * correct / total


def test(testloader, net, config, device, attack=''):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(
                tqdm(testloader, desc='{}-{}-{}'.format(config['dataset'], attack, config['model_name']), ncols=100)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100. * correct / total


def save_net(net, acc, path, file):
    state = {
        'net': net.module.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    torch.save(state, os.path.join(path, file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='AID')
    parser.add_argument('--model_name', type=str, default='shufflenet_v2_x2_0')
    parser.add_argument('--gpu', type=str, default='4')
    args = parser.parse_args()

    config = load_config('./config.yaml')
    config = updata_config(args, config)
    warnings_ignore()
    fix_seed(config['seed'])
    set_gpu(args.gpu)

    path = config['model_save_path']
    path_check(path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainloader, testloader = load_data(config[args.dataset]['path'], config[args.model_name]['batchsize'])
    net = load_model(config, args, device, pretrain=True, mode='train')

    criterion = nn.CrossEntropyLoss()

    save_freq = config[args.model_name]['save_freq']
    opt = config[args.model_name]['optimizer']
    if opt == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=config[args.model_name][opt]['lr'])

    for epoch in range(1, config[args.model_name]['epoch'] + 1):
        acc = train(epoch)
        if save_freq == 'None':
            pass
        elif epoch % save_freq == 0:
            path_process = os.path.join(path, args.model_name + '_process')
            path_check(path_process)
            file = str(epoch) + '.pth'
            save_net(net, acc, path_process, file)
    file = args.model_name + '.pth'
    save_net(net, acc, path, file)
