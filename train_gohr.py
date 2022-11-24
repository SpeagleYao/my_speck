import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import torch.optim.lr_scheduler as sch
from torch.utils.data import DataLoader, TensorDataset
from models import *
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Model Train')
parser.add_argument('--batch-size', type=int, default=5000,
                help='input batch size for training (default: 5000)')
parser.add_argument('--test-batch-size', type=int, default=100000,
                help='input batch size for testing (default: 100000)')
parser.add_argument('--epochs', type=int, default=20,
                help='number of epochs to train (default: 20)')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                help='weight decay (default: 1e-5)')
parser.add_argument('--base-lr', type=float, default=2e-3,
                help='learning rate (default: 2e-3)')
parser.add_argument('--max-lr', type=float, default=1e-4,
                help='learning rate (default: 1e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1,
                help='random seed (default: 1)')
args = parser.parse_args()

torch.manual_seed(args.seed)
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.manual_seed(args.seed)
else:
    print("No cuda participate.")

X_train = torch.as_tensor(np.load("./data/7r/train_data_7r.npy")).to(torch.float32)
Y_train = torch.as_tensor(np.load("./data/7r/train_label_7r.npy")).to(torch.float32)
X_test = torch.as_tensor(np.load("./data/7r/test_data_7r.npy")).to(torch.float32)
Y_test= torch.as_tensor(np.load("./data/7r/test_label_7r.npy")).to(torch.float32)
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=args.batch_size)
test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=args.test_batch_size)

model = ResNet_Gohr(blocks=10)
if args.cuda:
    model.cuda()

criterion = nn.MSELoss().cuda()
optimizer = opt.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
# scheduler = sch.CyclicLR(optimizer, base_lr = args.base_lr, max_lr = args.max_lr, step_size_up = 9, step_size_down = 1, cycle_momentum=False)
lambda1 = lambda epoch: (args.max_lr + (9 - epoch % 10)/9 * (args.base_lr - 0.0001))/args.base_lr
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data).squeeze()
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            tqdm.write('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    
    filename = './checkpoints/res_gohr_7r.pth'
    torch.save(model.state_dict(), filename)

def inference():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data).squeeze()
            test_loss += F.mse_loss(output, target, reduction='sum').item()
            correct += (output.ge(0.5) == target).sum().item()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * float(correct) / len(test_loader.dataset)))

for epoch in range(1, args.epochs + 1):
    train(epoch)
    inference()