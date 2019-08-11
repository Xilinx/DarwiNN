
import torch

def compute_ranks(x, device='cpu'):
    ranks = torch.zeros(len(x), dtype=torch.float, device=device)
    sort, ind = x.sort()
    for i in range(len(x)):
        ranks[ind[i].data] = i
    return ranks

def compute_centered_ranks(x, device='cpu'):
    centered = torch.zeros(len(x), dtype=torch.float, device=device)
    sort, ind = x.sort()
    for i in range(len(x)):
        centered[ind[i].data] = i
    centered = torch.div(centered, len(x) - 1)
    centered = centered - 0.5
    return centered
    
def compute_normalized_ranks(x, r=0.5, device='cpu'):
    n = len(x)
    ranks = compute_ranks(x) + 1
    incr = torch.arange(1,n+1, dtype=torch.float, device=device)
    rank_th_log=torch.log(torch.tensor([n*r+1], dtype=torch.float))
    den = torch.max(torch.tensor([0],dtype=torch.float),rank_th_log-torch.log(ranks))
    num = torch.sum(torch.max(torch.tensor([0],dtype=torch.float),rank_th_log-torch.log(incr)))
    ranks = torch.div(den,num) - 1/n
    return ranks