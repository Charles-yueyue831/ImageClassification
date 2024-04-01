# --*-- coding:utf-8 --*--
# @Author : 一只楚楚猫
# @File : main.py
# @Software : PyCharm

import os
import time
import importlib
import json
from collections import OrderedDict
import logging
import argparse
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.utils as utils

from dataloader import get_loader

try:
    from tensorboardX import SummaryWriter

    is_tensorboard_available = True
except Exception:
    is_tensorboard_available = False

torch.backends.cudnn.benchmark = True

logging.basicConfig(
    format='[%(asctime)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)  # 获取logger对象

global_step = 0


def str2bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise RuntimeError('Boolean value expected')


def parse_args():
    parser = argparse.ArgumentParser()

    # model config
    parser.add_argument('--depth', type=int, default=28)  # 神经网络的深度
    parser.add_argument('--base_channels', type=int, default=16)  # 通道数
    parser.add_argument('--widen_factor', type=int, default=10)  # wrn_k
    parser.add_argument('--dropout', type=float, default=0)  # Dropout

    # run config
    parser.add_argument('--outdir', type=str, default="./outputs")  # 输出路径
    parser.add_argument('--seed', type=int, default=17)  # 随机种子
    parser.add_argument('--num_workers', type=int, default=7)  # 数据加载的进程数

    # optim config
    parser.add_argument('--epochs', type=int, default=200)  # 训练轮数
    parser.add_argument('--batch_size', type=int, default=128)  # 批次大小
    parser.add_argument('--base_lr', type=float, default=0.1)  # 学习率
    parser.add_argument('--weight_decay', type=float, default=5e-4)  # 权值衰减系数
    parser.add_argument('--momentum', type=float, default=0.9)  # 动量系数
    parser.add_argument('--nesterov', type=str2bool, default=True)  # 是否使用Nesterov momentum
    parser.add_argument('--milestones', type=str, default='[60, 120, 160]')  # 学习率衰减的epoch数
    parser.add_argument('--lr_decay', type=float, default=0.2)  # 学习率衰减系数

    # TensorBoard
    """
    dest: parser对象的成员属性名称
        long option strings -- firstly
        then short option strings -
    """
    parser.add_argument(
        '--tensorboard', dest='tensorboard', default=True)

    args = parser.parse_args()
    if not is_tensorboard_available:
        args.tensorboard = False

    model_config = OrderedDict([
        ("architecture", "WRN"),
        ("depth", args.depth),
        ("base_channels", args.base_channels),
        ("widen_factor", args.widen_factor),
        ("dropout", args.dropout),
        ("input_shape", (1, 3, 32, 32)),
        ("n_classes", 10)
    ])

    optim_config = OrderedDict([
        ('epochs', args.epochs),
        ('batch_size', args.batch_size),
        ('base_lr', args.base_lr),
        ('weight_decay', args.weight_decay),
        ('momentum', args.momentum),
        ('nesterov', args.nesterov),
        ('milestones', json.loads(args.milestones)),
        ('lr_decay', args.lr_decay),
    ])

    data_config = OrderedDict([
        ('dataset', 'CIFAR10'),
    ])

    run_config = OrderedDict([
        ('seed', args.seed),
        ('outdir', args.outdir),
        ('num_workers', args.num_workers),
        ('tensorboard', args.tensorboard),
    ])

    config = OrderedDict([
        ('model_config', model_config),
        ('optim_config', optim_config),
        ('data_config', data_config),
        ('run_config', run_config),
    ])

    return config


def load_model(config):
    module = importlib.import_module(config["architecture"])
    WRN = getattr(module, 'WRN')
    return WRN(config)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, num):
        """
        计算accuracy和loss的平均值
        :param val: loss or accuracy
        :param num: 图像的数量
        :return:
        """
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def train(epoch, wrn, optimizer, criterion, train_loader, run_config, writer):
    global global_step

    logger.info('Train {}'.format(epoch))
    wrn.train()

    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    start = time.time()

    for step, (data, targets) in enumerate(train_loader):
        global_step += 1

        if run_config['tensorboard'] and step == 0:
            """
            make_grid的作用是将若干幅图像拼接成一幅图像
                padding: 子图像与子图像之间的pad有多宽
                normalize: 是否将像素值归一化到0-1
                scale_each: 是否将每个通道归一化
            """
            image = torchvision.utils.make_grid(
                data, normalize=True, scale_each=True)
            writer.add_image('Train/Image', image, epoch)

        data = data.cuda()
        targets = targets.cuda()
        optimizer.zero_grad()

        outputs = wrn(data)
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        predicted = torch.max(outputs, 1)[1]

        loss_ = loss.item()
        correct_ = predicted.eq(targets).sum().item()
        num = data.size()[0]  # 图像的数量

        accuracy = correct_ / num
        loss_meter.update(loss_, num)
        accuracy_meter.update(accuracy, num)

        if run_config['tensorboard']:
            """
            add_scalar(tag, scalar_value, global_step=None, ): 将我们所需要的数据保存在文件里面供可视化使用
                tag: 保存图的名称
                scalar_value: y轴数据
                global_step: x轴数据
            """
            writer.add_scalar('Train/RunningLoss', loss_, global_step)
            writer.add_scalar('Train/RunningAccuracy', accuracy, global_step)

        if step % 100 == 0:
            logger.info('Epoch {} Step {}/{} '
                        'Loss {:.4f} ({:.4f}) '
                        'Accuracy {:.4f} ({:.4f})'.format(
                epoch,
                step,
                len(train_loader),
                loss_meter.val,
                loss_meter.avg,
                accuracy_meter.val,
                accuracy_meter.avg,
            ))

        elapsed = time.time() - start
        logger.info('Elapsed {:.2f}'.format(elapsed))

        if run_config['tensorboard']:
            writer.add_scalar('Train/Loss', loss_meter.avg, epoch)
            writer.add_scalar('Train/Accuracy', accuracy_meter.avg, epoch)
            writer.add_scalar('Train/Time', elapsed, epoch)


def test(epoch, wrn, criterion, test_loader, run_config, writer):
    logger.info('Test {}'.format(epoch))
    wrn.eval()

    loss_meter = AverageMeter()
    correct_meter = AverageMeter()

    start = time.time()

    for step, (data, targets) in enumerate(test_loader):
        if run_config['tensorboard'] and epoch == 0 and step == 0:
            image = torchvision.utils.make_grid(
                data, normalize=True, scale_each=True)
            writer.add_image('Test/Image', image, epoch)

        data = data.cuda()
        targets = targets.cuda()

        with torch.no_grad():
            outputs = wrn(data)

        loss = criterion(outputs, targets)

        predicted = torch.max(outputs, 1)[1]

        loss_ = loss.item()
        correct_ = predicted.eq(targets).sum().item()
        num = data.size()[0]

        loss_meter.update(loss_, num)
        correct_meter.update(correct_, 1)

    accuracy = correct_meter.sum / len(test_loader.dataset)

    logger.info('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch, loss_meter.avg, accuracy))

    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))

    if run_config['tensorboard']:
        if epoch > 0:
            writer.add_scalar("Test/Loss", loss_meter.avg, epoch)
        writer.add_scalar("Test/Accuracy", accuracy, epoch)
        writer.add_scalar("Test/Elapsed", elapsed, epoch)

    return accuracy


def main():
    config = parse_args()

    """
    json.dumps(): 序列化，将python对象转换为json对象
        indent: 控制缩进格式
    json.loads(): 反序列化，将json对象转换为python对象
    json.dump(): 序列化到文件
    json.load(): 从文件中加载
    """
    logger.info(json.dumps(config, indent=2))

    run_config = config['run_config']
    optim_config = config['optim_config']

    # TensorBoard SummaryWriter
    writer = SummaryWriter() if run_config['tensorboard'] else None

    # set random seed
    seed = run_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    """
    torch.backends.cudnn.deterministic: 使用相同的算法和参数来生成随机数
    torch.cuda.manual_seed_all: 设置所有GPU的种子
    """
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

    # create output directory
    outdir = run_config['outdir']
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # save config as json file in output directory
    outpath = os.path.join(outdir, 'config.json')
    with open(outpath, 'w') as fout:
        json.dump(config, fout, indent=2)

    # data loaders
    train_loader, test_loader = get_loader(optim_config['batch_size'],
                                           run_config['num_workers'])

    # model
    wrn = load_model(config['model_config'])
    wrn.cuda()

    n_params = sum([param.view(-1).size()[0] for param in wrn.parameters()])
    logger.info('n_params: {}'.format(n_params))

    """
    size_average（该参数不建议使用，后续版本可能被废弃），该参数指定loss是否在一个Batch内平均，即是否除以N
    """
    criterion = nn.CrossEntropyLoss(size_average=True)

    # optimizer
    optimizer = torch.optim.SGD(
        wrn.parameters(),
        lr=optim_config['base_lr'],
        momentum=optim_config['momentum'],
        weight_decay=optim_config['weight_decay'],
        nesterov=optim_config['nesterov'])

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=optim_config['milestones'],
                                                     gamma=optim_config['lr_decay'])

    # run test before start training
    test(epoch=0, wrn=wrn, criterion=criterion, test_loader=test_loader, run_config=run_config, writer=writer)

    for epoch in range(optim_config["epochs"] + 1):
        scheduler.step()

        train(epoch=epoch, wrn=wrn, criterion=criterion, train_loader=train_loader, optimizer=optimizer,
              run_config=run_config, writer=writer)
        accuracy = test(epoch=epoch, wrn=wrn, criterion=criterion, test_loader=test_loader, run_config=run_config,
                        writer=writer)

        state = OrderedDict([
            ("config", config),
            ("state_dict", wrn.state_dict()),
            ("optimizer", optimizer.state_dict()),
            ("epoch", epoch),
            ("accuracy", accuracy)
        ])

        wrn_path = os.path.join(outdir, "wrn_state.pth")
        torch.save(state, wrn_path)

    if run_config["tensorboard"]:
        writer_path = os.path.join(outdir, "wrn_tensorboard.json")
        writer.export_scalars_to_json(writer_path)


if __name__ == "__main__":
    main()