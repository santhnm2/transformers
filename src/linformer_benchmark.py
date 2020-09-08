import numpy as np
import time
import torch

num_trials = 10

h = 12
d = 64
k = 128

def experiment(b, h, n, d, k):
    Q = torch.randn((b, h, n, d)).cuda()
    K = torch.randn((b, h, n, d)).cuda()
    V = torch.randn((b, h, n, d)).cuda()
    E = torch.randn((n, k)).cuda()
    softmax = torch.nn.Softmax(dim=-1)

    mask = torch.randint(low=0, high=1, size=(b, n)).cuda()
    mask_reshp = (b, 1, 1, n)
    times = []
    for i in range(num_trials):
        start = time.time()
        scores = torch.matmul(Q, K.transpose(2, 3))
        mask_ = (mask == 0).view(mask_reshp).expand_as(scores)
        scores.masked_fill_(mask_, -1e10)
        weights = softmax(scores)
        torch.matmul(weights, V)
        torch.cuda.synchronize()
        times.append(time.time() - start)
        del mask_
    original_time = np.median(times) * 1000

    torch.cuda.empty_cache()

    mask_reshp = (b, 1, 1, k)
    times = []
    for i in range(num_trials):
        start = time.time()
        linformer_K = torch.einsum('bhnd,nk->bhkd', K, E)
        linformer_V = torch.einsum('bhnd,nk->bhkd', V, E)
        scores = torch.matmul(Q, linformer_K.transpose(2, 3))
        mask_ = mask[:, :k]
        mask_ = (mask_ == 0).view(mask_reshp).expand_as(scores)
        scores.masked_fill_(mask_, -1e10)
        weights = softmax(scores)
        torch.matmul(weights, linformer_V)
        torch.cuda.synchronize()
        times.append(time.time() - start)
        del mask_
    linformer_time = np.median(times) * 1000

    del Q
    del K
    del V
    del E

    return original_time, linformer_time

print('Batch size|n|k|Original op|Linformer op|Original runtime (ms)|Linformer runtime (ms)')
for b in [1, 4, 16, 32]:
    for n in [512, 1024, 2048, 4096]:
        try:
            original_time, linformer_time = experiment(b, h, n, d, k)
        except Exception as e:
            original_time = -1
            linformer_time = -1
        original_shape = '[%d,%d,%d,%d] x [%d,%d,%d,%d]' % (b, h, n, n, b, h, n, d)
        linformer_shape = '[%d,%d,%d,%d] x [%d,%d,%d,%d]' % (b, h, n, k, b, h, k, d)
        print('%d|%d|%d|%s|%s|%f|%f' % (b, n, k, original_shape, linformer_shape, original_time, linformer_time))
