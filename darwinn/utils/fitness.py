#  Copyright (c) 2019, Xilinx
#  All rights reserved.
#  
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#  
#  1. Redistributions of source code must retain the above copyright notice, this
#     list of conditions and the following disclaimer.
#  
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#  
#  3. Neither the name of the copyright holder nor the names of its
#     contributors may be used to endorse or promote products derived from
#     this software without specific prior written permission.
#  
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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