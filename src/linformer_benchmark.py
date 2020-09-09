import numpy as np
import time
import torch

num_trials = 10

h = 12
d = 64
k = 256

def _transpose_for_scores(x, h):
    x = x.reshape(x.shape[0], x.shape[1], h, x.shape[-1] // h)
    x = x.permute(0, 2, 1, 3)
    return x

def experiment(b, h, n, d, k):
    x = torch.randn((b, n, d * h)).cuda()
    query_layer = torch.nn.Linear(d * h, d * h).cuda()
    key_layer = torch.nn.Linear(d * h, d * h).cuda()
    value_layer = torch.nn.Linear(d * h, d * h).cuda()
    E = torch.randn((n, k)).cuda()
    softmax = torch.nn.Softmax(dim=-1).cuda()

    times = []

    for i in range(num_trials):
        start = time.time()
        Q = query_layer(x)
        K = key_layer(x)
        V = value_layer(x)

        Q = _transpose_for_scores(Q, h)
        K = _transpose_for_scores(K, h)
        V = _transpose_for_scores(V, h)

        scores = torch.matmul(Q, K.transpose(2, 3))
        scores = torch.div(scores, np.sqrt(d))
        weights = softmax(scores)
        torch.matmul(weights, V)
        torch.cuda.synchronize()
        times.append(time.time() - start)
    original_time = np.median(times) * 1000

    torch.cuda.empty_cache()

    mask_reshp = (b, 1, 1, k)
    times = []
    for i in range(num_trials):
        start = time.time()
        Q = query_layer(x)
        K = key_layer(x)
        V = value_layer(x)

        Q = _transpose_for_scores(Q, h)
        K = _transpose_for_scores(K, h)
        V = _transpose_for_scores(V, h)

        linformer_K = torch.einsum('bhnd,nk->bhkd', K, E)
        linformer_V = torch.einsum('bhnd,nk->bhkd', V, E)

        scores = torch.matmul(Q, linformer_K.transpose(2, 3))
        weights = softmax(scores)
        torch.matmul(weights, linformer_V)
        torch.cuda.synchronize()
        times.append(time.time() - start)
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
