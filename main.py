'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import matplotlib.pyplot as plt

from models import *
# from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda:0'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt_rmsprop.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    (train_loss_list, train_acc_list, test_loss_list, test_acc_list) = checkpoint['train_history']

criterion = nn.CrossEntropyLoss()
# mse = nn.MSELoss(reduce=True, size_average=True)
# nll = nn.NLLLoss(weight=None, ignore_index=-100, reduce=None, reduction='mean')
# optimizer = optim.SGD(net.parameters(), lr=args.lr,
#                       momentum=0.9, weight_decay=5e-4)
#optimizer = optim.Adam(net.parameters(), lr=args.lr)
optimizer = optim.RMSprop(net.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_index = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        # loss = nll(outputs, targets)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_index = batch_idx
        # train_loss += loss.item()
        train_loss = loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print("\n")
    train_loss_list.append(train_loss/(batch_index+1))
    train_acc_list.append(100.*correct/total)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    batch_index = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            # loss = nll(outputs, targets)
            batch_index = batch_idx
            # test_loss += loss.item()
            test_loss = loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            print("\n")
    test_loss_list.append(test_loss/(batch_index+1))
    test_acc_list.append(100.*correct/total)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'train_history': (train_loss_list, train_acc_list, test_loss_list, test_acc_list)
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_rmsprop.pth')
        best_acc = acc


def visualize(train_loss, train_acc, test_loss, test_acc, label):
    length = len(train_acc)
    plt.subplot(2, 2, 1)
    plt.plot(range(length), train_acc, label=label)
    plt.title('Train accuracy vs. epoches')
    plt.ylabel('Train accuracy')
    plt.xlabel('epoch')
    plt.subplot(2, 2, 2)
    plt.plot(range(length), train_loss, label=label)
    plt.title('Train loss vs. epoches')
    plt.ylabel('Train loss')
    plt.xlabel('epoch')
    plt.subplot(2, 2, 3)
    plt.plot(range(length), test_acc, label=label)
    plt.title('Test accuracy vs. epoches')
    plt.ylabel('Test accuracy')
    plt.xlabel('epoch')
    plt.subplot(2, 2, 4)
    plt.plot(range(length), test_loss, label=label)
    plt.title('Test loss vs. epoches')
    plt.ylabel('Test loss')
    plt.xlabel('epoch')


def lr_plot():
    lr = ["_lr_0.01", "", "_lr_0.3", "_lr_0.5"]
    label = ["lr = 0.01", "lr = 0.1", "lr = 0.3", "lr = 0.5"]
    checkpoints = []
    best_acc_list = []
    (train_loss_lists, train_acc_lists, test_loss_lists, test_acc_lists) = \
        ([[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []])
    for i in range(4):
        checkpoints.append(torch.load('./checkpoint/ckpt' + lr[i] + '.pth'))
        best_acc_list.append(checkpoints[i]['acc'])
        (train_loss_lists[i], train_acc_lists[i], test_loss_lists[i], test_acc_lists[i]) \
            = checkpoints[i]['train_history']
        visualize(train_loss_lists[i], train_acc_lists[i], test_loss_lists[i], test_acc_lists[i], label[i])
    print(best_acc_list)


def optimizer_plot():
    optimizers = ["", "_adam", "_rmsprop"]
    label = ["SGD", "Adam", "RMSProp"]
    checkpoints = []
    best_acc_list = []
    (train_loss_lists, train_acc_lists, test_loss_lists, test_acc_lists) = \
        ([[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []])
    for i in range(3):
        checkpoints.append(torch.load('./checkpoint/ckpt' + optimizers[i] + '.pth'))
        best_acc_list.append(checkpoints[i]['acc'])
        (train_loss_lists[i], train_acc_lists[i], test_loss_lists[i], test_acc_lists[i]) \
            = checkpoints[i]['train_history']
        visualize(train_loss_lists[i], train_acc_lists[i], test_loss_lists[i], test_acc_lists[i], label[i])
    print(best_acc_list)


def scheduler_plot():
    schedulers = ["", "_no_scheduler"]
    label = ["lr scheduler", "no scheduler"]
    checkpoints = []
    best_acc_list = []
    (train_loss_lists, train_acc_lists, test_loss_lists, test_acc_lists) = \
        ([[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []])
    for i in range(2):
        checkpoints.append(torch.load('./checkpoint/ckpt' + schedulers[i] + '.pth'))
        best_acc_list.append(checkpoints[i]['acc'])
        (train_loss_lists[i], train_acc_lists[i], test_loss_lists[i], test_acc_lists[i]) \
            = checkpoints[i]['train_history']
        visualize(train_loss_lists[i], train_acc_lists[i], test_loss_lists[i], test_acc_lists[i], label[i])
    print(best_acc_list)


if __name__=='__main__':
    # epochs = start_epoch
    # for epoch in range(start_epoch, start_epoch+200):
    #     epochs = epoch
    #     train(epoch)
    #     test(epoch)
    #     scheduler.step()

    # visualize(116)
    scheduler_plot()
    plt.legend()
    plt.tight_layout()
    plt.show()
