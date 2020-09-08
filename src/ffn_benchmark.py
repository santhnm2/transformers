import numpy as np
import time
import torch

num_trials = 10

h = 12
d = 64
k = 3072

def experiment(b, h, n, d, k):
    x = torch.randn((b, n, d * h)).cuda()
    i = torch.randn((d * h, k)).cuda()
    o = torch.randn((k, d * h)).cuda()

    runtimes = []
    for _ in range(num_trials):
        start = time.time()
        torch.matmul(torch.matmul(x, i), o)
        torch.cuda.synchronize()
        runtimes.append(time.time() - start)
    return np.median(runtimes)

print('Batch size,n,Runtime (ms)')
for b in [1, 4, 16, 32]:
    for n in [512, 1024, 2048, 4096]:
        runtime = experiment(b, h, n, d, k)
        print('%d,%d,%f' % (b, n, runtime * 1000))
