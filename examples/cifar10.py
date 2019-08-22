import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from darwinn.utils.environment import DarwiNNEnvironment
from darwinn.optimizers.dnn import OpenAIESOptimizer
from darwinn.optimizers.dnn import GAOptimizer
import argparse
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
    
class c10q(nn.Module):
    def __init__(self):
        super(c10q, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1,padding=2)
        self.fc1 = nn.Linear(64*4*4,64)
        self.fc2 = nn.Linear(64,10)

    def forward(self, x):
        out = self.conv1(x)
        out = F.max_pool2d(out,kernel_size=3, stride=2, ceil_mode=True)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = F.avg_pool2d(out,kernel_size=3, stride=2, ceil_mode=True)
        out = self.conv3(out)
        out = F.relu(out)
        out = F.avg_pool2d(out,kernel_size=3, stride=2, ceil_mode=True)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out

class NiN(nn.Module):
    def __init__(self):
        super(NiN, self).__init__()
        self.classifier = nn.Sequential(
                nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(160,  96, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Dropout(0.5),

                nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                nn.Dropout(0.5),

                nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(192,  10, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=8, stride=1, padding=0),

                )

    def forward(self, x):
        x = self.classifier(x)
        x = x.view(x.size(0), 10)
        return x

class CIF_300K(nn.Module):
    def __init__(self):
        super(CIF_300K, self).__init__()
        self.conv1d = nn.Conv2d(  3,   3, kernel_size=3, stride=1, padding=1, groups=3, bias=False)
        self.conv1p = nn.Conv2d(  3,  64, kernel_size=1, stride=1, bias=False)
        self.conv1b = nn.BatchNorm2d(64)
        self.conv2d = nn.Conv2d( 64,  64, kernel_size=3, stride=1, padding=1, groups=64, bias=False)
        self.conv2p = nn.Conv2d( 64,  64, kernel_size=1, stride=1, bias=False)
        self.conv2b = nn.BatchNorm2d(64)
        self.conv3d = nn.Conv2d( 64,  64, kernel_size=3, stride=2, padding=1, groups=64, bias=False)
        self.conv3p = nn.Conv2d( 64, 128, kernel_size=1, stride=1, bias=False)
        self.conv3b = nn.BatchNorm2d(128)
        self.conv4d = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False)
        self.conv4p = nn.Conv2d(128, 128, kernel_size=1, stride=1, bias=False)
        self.conv4b = nn.BatchNorm2d(128)
        self.conv5d = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False)
        self.conv5p = nn.Conv2d(128, 128, kernel_size=1, stride=1, bias=False)
        self.conv5b = nn.BatchNorm2d(128)
        self.conv6d = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, groups=128, bias=False)
        self.conv6p = nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False)
        self.conv6b = nn.BatchNorm2d(256)
        self.conv7d = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=256, bias=False)
        self.conv7p = nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False)
        self.conv7b = nn.BatchNorm2d(256)
        self.conv8d = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=256, bias=False)
        self.conv8p = nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False)
        self.conv8b = nn.BatchNorm2d(256)
        self.conv9d = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, groups=256, bias=False)
        self.conv9p = nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False)
        self.conv9b = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512,10, bias=True)

    def convstack(self, x):
        out = self.conv1d(x)
        out = self.conv1p(out)
        out = self.conv1b(out)
        out = F.relu(out)
        out = self.conv2d(out)
        out = self.conv2p(out)
        out = self.conv2b(out)
        out = F.relu(out)
        out = self.conv3d(out)
        out = self.conv3p(out)
        out = self.conv3b(out)
        out = F.relu(out)
        out = self.conv4d(out)
        out = self.conv4p(out)
        out = self.conv4b(out)
        out = F.relu(out)
        out = self.conv5d(out)
        out = self.conv5p(out)
        out = self.conv5b(out)
        out = F.relu(out)
        out = self.conv6d(out)
        out = self.conv6p(out)
        out = self.conv6b(out)
        out = F.relu(out)
        out = self.conv7d(out)
        out = self.conv7p(out)
        out = self.conv7b(out)
        out = F.relu(out)
        out = self.conv8d(out)
        out = self.conv8p(out)
        out = self.conv8b(out)
        out = F.relu(out)
        out = self.conv9d(out)
        out = self.conv9p(out)
        out = self.conv9b(out)
        out = F.relu(out)
        return out

    def forward(self, x):
        out = self.convstack(x)
        out = F.avg_pool2d(out,kernel_size=4, stride=1, ceil_mode=True)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
        
class CIF_900K(CIF_300K):
    def __init__(self):
        super(CIF_900K, self).__init__()
        self.conv10d = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512, bias=False)
        self.conv10p = nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False)
        self.conv10b = nn.BatchNorm2d(512)
        self.conv11d = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512, bias=False)
        self.conv11p = nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False)
        self.conv11b = nn.BatchNorm2d(512)

    def forward(self, x):
        out = self.convstack(x)
        out = self.conv10d(out)
        out = self.conv10p(out)
        out = self.conv10b(out)
        out = F.relu(out)
        out = self.conv11d(out)
        out = self.conv11p(out)
        out = self.conv11b(out)
        out = F.relu(out)
        out = F.avg_pool2d(out,kernel_size=4, stride=1, ceil_mode=True)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class CIF_8M(nn.Module):
    def __init__(self):
        super(CIF_8M, self).__init__()
        self.conv1 = nn.Conv2d(  3,  64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1b = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d( 64,  64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2b = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d( 64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3b = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4b = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5b = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv6b = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv7b = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv8b = nn.BatchNorm2d(256)
        self.conv9 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv9b = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv10b = nn.BatchNorm2d(512)
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv11b = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512,10, bias=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv1b(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.conv2b(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.conv3b(out)
        out = F.relu(out)
        out = self.conv4(out)
        out = self.conv4b(out)
        out = F.relu(out)
        out = self.conv5(out)
        out = self.conv5b(out)
        out = F.relu(out)
        out = self.conv6(out)
        out = self.conv6b(out)
        out = F.relu(out)
        out = self.conv7(out)
        out = self.conv7b(out)
        out = F.relu(out)
        out = self.conv8(out)
        out = self.conv8b(out)
        out = F.relu(out)
        out = self.conv9(out)
        out = self.conv9b(out)
        out = F.relu(out)
        out = self.conv10(out)
        out = self.conv10b(out)
        out = F.relu(out)
        out = self.conv11(out)
        out = self.conv11b(out)
        out = F.relu(out)
        out = F.avg_pool2d(out,kernel_size=4, stride=1, ceil_mode=True)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def train(epoch, train_loader, model, criterion, optimizer, args):
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        if args.backprop:
            optimizer.zero_grad()
            results = model(data)
            loss = criterion(results, target)
            loss.backward()
            optimizer.step()
        else:
            optimizer.step(data, target)
        if batch_idx % args.log_interval == 0 and args.verbose:
            if args.backprop:
                loss_val = loss.item()
            else:
                loss_val = optimizer.get_loss()
            print('Train Epoch: {} (batch {})\tLoss: {:.6f}'.format(epoch, batch_idx, loss_val))

def test(test_loader, model, criterion, optimizer, args):
    test_loss = 0.
    test_accuracy = 0.
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        if args.backprop:
            with torch.no_grad():
                output = model(data)
                loss = criterion(output,target).item()
        else:
            output = optimizer.eval_theta(data, target)
            loss = optimizer.get_loss()
        # sum up batch loss
        test_loss += loss
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()

    # use test_sampler to determine the number of examples in
    # this worker's partition.
    test_loss /= len(test_loader)
    test_accuracy /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(test_loss, 100. * test_accuracy))

if __name__ == "__main__":
    
    # Training settings
    parser = argparse.ArgumentParser(description='Neuroevolution PyTorch CIFAR10 Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-test', action='store_true', default=False,
                        help='disables testing')
    dist_mode = parser.add_mutually_exclusive_group()
    dist_mode.add_argument('--backprop', action='store_true', default=False,
                        help='performs training with Backpropagation')
    dist_mode.add_argument('--ddp', action='store_true', default=False,
                        help='performs Distributed Data-Parallel evolution')
    dist_mode.add_argument('--semi-updates', action='store_true', default=False,
                        help='performs Semi-Updates in OpenAI-ES')
    dist_mode.add_argument('--orthogonal-updates', action='store_true', default=False,
                        help='performs Orthogonal Updates in OpenAI-ES')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--topology', type=str, choices=['LeNet','c10q','NiN','CIF_300K','CIF_900K','CIF_8M'],
	                    default='LeNet', help='NN topology (default: LeNet)')
    parser.add_argument('--popsize', type=int, default=100, metavar='N',
                        help='population size (default: 100)')
    parser.add_argument('--noise-dist', type=str, default="Gaussian",
                        help='noise distribution (default: Gaussian)')
    parser.add_argument('--sigma', type=float, default=0.01, metavar='S',
                        help='noise variance (default: 0.01)')
    parser.add_argument('--sampling', type=str, default="Antithetic",
                        help='sampling strategy (default: Antithetic)')
    parser.add_argument('--ne-opt',default='OpenAI-ES',choices=['OpenAI-ES', 'GA'],
                        help='choose which neuroevolution optimizer to use')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='enables printing loss during training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    env = DarwiNNEnvironment(args.cuda)
    
    dataset_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = datasets.CIFAR10('CIFAR10_data_'+str(env.rank), train=True, download=True, transform=dataset_transform)
    
    if args.backprop:
        args.ddp = True
    
    if args.ddp:
        train_ddp_sampler = DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_ddp_sampler, batch_size=args.batch_size, **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    
    test_dataset = datasets.CIFAR10('CIFAR10_data_'+str(env.rank), train=False, download=False, transform=dataset_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    loss_criterion = nn.CrossEntropyLoss()
    
    if args.topology == 'LeNet':
        model = LeNet()
    elif args.topology == 'c10q':
        model = c10q()
    elif args.topology == 'NiN':
        model = NiN()
    elif args.topology == 'CIF_300K':
        model = CIF_300K()
    elif args.topology == 'CIF_900K':
        model = CIF_900K()
    elif args.topology == 'CIF_8M':
        model = CIF_8M()

    if args.backprop:
        if args.cuda:
            model = DDP(model.cuda(), device_ids=[torch.cuda.current_device()], output_device=torch.cuda.current_device())
        else:
            model = DDP(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.ne_opt == 'OpenAI-ES':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        #wrap optimizer into a OpenAI-ES optimizer
        optimizer = OpenAIESOptimizer(env, model, loss_criterion, optimizer, sigma=args.sigma, popsize=args.popsize, distribution=args.noise_dist, sampling=args.sampling, data_parallel=args.ddp, semi_updates=args.semi_updates, orthogonal_updates=args.orthogonal_updates)
    else:
        optimizer = GAOptimizer(env, model, loss_criterion, sigma=args.sigma, popsize=args.popsize, data_parallel=args.ddp)

    for epoch in range(1, args.epochs + 1):
        train(epoch, train_loader, model, loss_criterion, optimizer, args)
        if (env.rank == 0 or args.backprop) and not args.no_test:
            test(test_loader, model, loss_criterion, optimizer, args)
