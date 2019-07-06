import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import darwinn as dwn
import argparse

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

def train(epoch, train_loader, ne_optimizer, args):
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        ne_optimizer.eval_fitness(data, target)
        ne_optimizer.step()#no backward pass, adapt instead of step
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} (batch {})\tLoss: {:.6f}'.format(epoch, batch_idx, ne_optimizer.get_loss()))

def test(test_loader, ne_optimizer, args):
    test_loss = 0.
    test_accuracy = 0.
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = ne_optimizer.eval_theta(data, target)
        # sum up batch loss
        test_loss += ne_optimizer.get_loss()
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
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--topology', type=str, choices=['LeNet','c10q','NiN'],
	                    default='LeNet', help='NN topology (default: LeNet)')
    parser.add_argument('--popsize', type=int, default=100, metavar='N',
                        help='population size (default: 100)')
    parser.add_argument('--noise-dist', type=str, default="Gaussian",
                        help='noise distribution (default: Gaussian)')
    parser.add_argument('--sigma', type=float, default=0.01, metavar='S',
                        help='noise variance (default: 0.01)')
    parser.add_argument('--sampling', type=str, default="Antithetic",
                        help='sampling strategy (default: Antithetic)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    env = dwn.DarwiNNEnvironment(args.cuda)
    
    dataset_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = datasets.CIFAR10('CIFAR10_data_'+str(env.rank), train=True, download=True, transform=dataset_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_dataset = datasets.CIFAR10('CIFAR10_data_'+str(env.rank), train=False, download=False, transform=dataset_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    loss_criterion = nn.CrossEntropyLoss()
    
    if args.topology == 'LeNet':
        model = LeNet()
    elif args.topology == 'c10q':
        model = c10q()
    elif args.topology == 'NiN':
        model = NiN()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #wrap optimizer into a OpenAI-ES optimizer
    ne_optimizer = dwn.OpenAIESOptimizer(env, model, loss_criterion, optimizer, sigma=args.sigma, popsize=args.popsize, distribution=args.noise_dist, sampling=args.sampling)
    
    for epoch in range(1, args.epochs + 1):
        train(epoch, train_loader, ne_optimizer, args)
        if env.rank == 0:
            test(test_loader, ne_optimizer, args)

