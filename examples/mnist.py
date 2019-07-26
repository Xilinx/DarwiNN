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

class Conv2(nn.Module):
  def __init__(self):
    super(Conv2, self).__init__()
    self.num_filter1 = 8
    self.num_filter2 = 16
    self.num_padding = 2
    # input is 28x28
    # padding=2 for same padding
    self.conv1 = nn.Conv2d(1, self.num_filter1, 5, padding=self.num_padding)
    # feature map size is 14*14 by pooling
    # padding=2 for same padding
    self.conv2 = nn.Conv2d(self.num_filter1, self.num_filter2, 5, padding=self.num_padding)
    # feature map size is 7*7 by pooling
    self.fc = nn.Linear(self.num_filter2*7*7, 10)

  def forward(self, x):
    x = F.max_pool2d(F.relu(self.conv1(x)), 2)
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = x.view(-1, self.num_filter2*7*7)   # reshape Variable
    x = self.fc(x)
    return F.log_softmax(x, dim=0)

#network used in arxiv 1712.06564 and 1906.03139
class MNIST_3M(nn.Module):
  def __init__(self):
    super(MNIST_3M, self).__init__()
    self.num_filter1 = 32
    self.num_filter2 = 64
    self.num_padding = 2
    # input is 28x28
    # padding=2 for same padding
    self.conv1 = nn.Conv2d(1, self.num_filter1, 5, padding=self.num_padding)
    # feature map size is 14*14 by pooling
    # padding=2 for same padding
    self.conv2 = nn.Conv2d(self.num_filter1, self.num_filter2, 5, padding=self.num_padding)
    # feature map size is 7*7 by pooling
    self.fc1 = nn.Linear(self.num_filter2*7*7, 1024)
    self.fc2 = nn.Linear(1024, 10)

  def forward(self, x):
    x = F.max_pool2d(F.relu(self.conv1(x)), 2)
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = x.view(-1, self.num_filter2*7*7)   # reshape Variable
    x = self.fc1(x)
    x = self.fc2(x)
    return F.log_softmax(x, dim=0)

    
def train(epoch, train_loader, ne_optimizer, args):
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        ne_optimizer.eval_fitness(data, target)
        ne_optimizer.step()#no backward pass, adapt instead of step
        if batch_idx % args.log_interval == 0 and args.verbose:
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
    parser = argparse.ArgumentParser(description='Neuroevolution PyTorch MNIST Example')
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
    parser.add_argument('--ddp', action='store_true', default=False,
                        help='performs Distributed Data-Parallel evolution')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--popsize', type=int, default=100, metavar='N',
                        help='population size (default: 100)')
    parser.add_argument('--noise-dist', type=str, default="Gaussian",
                        help='noise distribution (default: Gaussian)')
    parser.add_argument('--sigma', type=float, default=0.01, metavar='S',
                        help='noise variance (default: 0.01)')
    parser.add_argument('--sampling', type=str, default="Antithetic",
                        help='sampling strategy (default: Antithetic)')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='enables printing loss during training')
    parser.add_argument('--ne-opt',default='OpenAI-ES',choices=['OpenAI-ES', 'GA'],
                        help='choose which neuroevolution optimizer to use')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    env = DarwiNNEnvironment(args.cuda)

    dataset_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('MNIST_data_'+str(env.rank), train=True, download=True, transform=dataset_transform)
    
    if args.ddp:
        train_ddp_sampler = DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_ddp_sampler, batch_size=args.batch_size, **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    
    test_dataset = datasets.MNIST('MNIST_data_'+str(env.rank), train=False, download=False, transform=dataset_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    loss_criterion = F.nll_loss
    
    
    model = Conv2()

    if args.ne_opt == 'OpenAI-ES':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        #wrap optimizer into a OpenAI-ES optimizer
        ne_optimizer = OpenAIESOptimizer(env, model, loss_criterion, optimizer, sigma=args.sigma, popsize=args.popsize, distribution=args.noise_dist, sampling=args.sampling, data_parallel=args.ddp)
    else:
        ne_optimizer = GAOptimizer(env, model, loss_criterion, sigma=args.sigma, popsize=args.popsize, data_parallel=args.ddp)
    
    for epoch in range(1, args.epochs + 1):
        train(epoch, train_loader, ne_optimizer, args)
        if env.rank == 0 and not args.no_test:
            test(test_loader, ne_optimizer, args)

