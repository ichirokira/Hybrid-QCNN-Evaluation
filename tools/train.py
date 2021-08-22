import numpy as np
import matplotlib.pyplot as plt
import time
import os
import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from backbones.hybrid_qcnn import Hybrid_RQCNN
from backbones.cnn import CNN
from tools.test import evaluation
from argparse import ArgumentParser

args = ArgumentParser()
args.add_argument('--method', '-m', default='random', type=str, help='Type of model [classical, structure, random]')
args.add_argument('--batch_size', '-b', default=1, type=int, help='Batch size of your dataset')
args.add_argument('--num_epoches', default=10, type=int, help='Number of Epoches')
args.add_argument('--num_iter', default=500, type=int, help='Number of iter per epoch')
args.add_argument('--n_test', default=5, type=int, help='Interval of evaluation')
args.add_argument('--save_log', default='tools/eval_stats/random', type=int, help='Saving Log file dir')
args.add_argument('--save_pretrained', default='tools/eval_stats/random', type=int, help='Saving pretrained file dir')
ap = args.parse_args()


X_train = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
train_loader = torch.utils.data.DataLoader(X_train, batch_size=ap.batch_size, shuffle=True)

X_test = datasets.MNIST(root='./data', train=False, download=True,
                        transform=transforms.Compose([transforms.ToTensor()]))
test_loader = torch.utils.data.DataLoader(X_test, batch_size=ap.batch_size, shuffle=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print("[INFO] Using {} method".format(ap.method))
if ap.method == 'classical':
    net = CNN(kernel_size=2, depth=4, num_classes=10)
else:
    net = Hybrid_RQCNN(kernel_size=2, depth=4, circuit_layers=1, device=device, method=ap.method, num_classes=10).to(device)
loss_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1.e-3)


if not os.path.exists(ap.save_log):
    os.makedirs('tools/eval_stats')


print("[INFO] Beginning Training")

for epoch in range(ap.num_epoches):
    epoch_time = time.time()
    correct = 0
    total = 0
    avg_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        if i==ap.num_iter:
            break
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = net(images)
        _, predicted = torch.max(output, 1)
        loss = loss_fun(output, labels).to(device)

        avg_loss += loss.item()
        correct += (predicted==labels).sum()
        total += labels.size(0)
        loss.backward()
        optimizer.step()
        #training_time = (time.time() - epoch_time)
        print("---Epoch %s took %s seconds ---" % (epoch + 1, (time.time() - epoch_time)))
        if (i+1) % (ap.num_iter // ap.n_test) == 0:
            print('[INFO] Epoch [%d/%d], Step[%d/%d]' % (epoch+1, ap.num_epoches, i+1, ap.num_iter))
            evaluation(net, test_loader, loss_fun, device, epoch, ap.save_log)
    avg_loss = avg_loss/ap.num_iter

    with open(os.path.join(ap.save_log, 'log_training.csv'), 'a') as f:
        f.write('%d, %.4f, %.4f\n' % (epoch, (100.0 * correct) / (total + 1), avg_loss))

    print('[INFO] Saving checkpoint for epoch', epoch+1)
    state = {
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, os.path.join(ap.save_pretrained,'/Hybrid_QCNN_epoch_{}.pt'.format(epoch+1)))



