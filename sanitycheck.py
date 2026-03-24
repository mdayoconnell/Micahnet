# Created by Micah
# Date: 12/11/25
# Time: 7:26 PM
# Project: MicahNet
# File: DONOTSKIP_Sanity_Check.py


import torch, time

device = "mps"
x = torch.randn(2000, 2000, device=device)

start = time.time()
for a in range(100):
    print(a)
    y = x @ x
torch.mps.synchronize()
print("Time:", time.time() - start)