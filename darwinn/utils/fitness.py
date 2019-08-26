
import torch

def compute_ranks(x, device='cpu'):
    ranks = torch.empty(len(x), dtype=torch.float, device=device)
    sort, ind = x.sort()
    ranks[ind] = torch.arange(len(x),dtype=torch.float,device=device)
    return ranks

def compute_centered_ranks(x, device='cpu'):
    centered = compute_ranks(x, device=device)
    centered /= (len(x) - 1)
    centered -= 0.5
    return centered
    
def compute_normalized_ranks(x, r=0.5, device='cpu'):
    n = len(x)
    ranks = compute_ranks(x,device=device) + 1
    incr = torch.arange(1,n+1, dtype=torch.float, device=device)
    rank_th_log=torch.log(torch.tensor([n*r+1], dtype=torch.float, device=device))
    den = torch.max(torch.tensor([0],dtype=torch.float,device=device),rank_th_log-torch.log(ranks))
    num = torch.sum(torch.max(torch.tensor([0],dtype=torch.float,device=device),rank_th_log-torch.log(incr)))
    ranks = torch.div(den,num) - 1/n
    return ranks