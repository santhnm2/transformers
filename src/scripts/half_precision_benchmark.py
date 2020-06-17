import argparse
import copy
import numpy as np
import time
import torch

#torch.backends.cudnn.benchmark = True

class HalfPrecisionBenchmark:
    def __init__(self):
        self._full_precision_layer = None
        self._half_precision_layer = None
        self._data = None

    def _profile(self, num_warmup_trials, num_trials, layer,
                half_precision=False):
        data = self._data

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Warm up.
        for i in range(num_warmup_trials):
            if half_precision:
                layer(data.half())
            else:
                layer(data)

        # Measure runtime.
        runtimes = []
        if half_precision:
            input_cast_times = []
            output_cast_times = []
        for i in range(num_trials):
            torch.cuda.synchronize()
            if half_precision:
                start_time = time.time()
                data = self._data.half()
                input_cast_times.append(time.time() - start_time)
            start_time = time.time()
            output = layer(data)
            runtimes.append(time.time() - start_time)
            if half_precision:
                start_time = time.time()
                output = output.half()
                output_cast_times.append(time.time() - start_time)

        if half_precision:
            return (np.mean(input_cast_times) * 1e6, np.mean(runtimes) * 1e6,
                    np.mean(output_cast_times) * 1e6)
        return np.mean(runtimes) * 1e6

    def profile(self, num_warmup_trials, num_trials):
        half_precision_layer = copy.deepcopy(self._layer)
        half_precision_layer.to('cuda')
        half_precision_layer.half()
        full_precision_runtime = self._profile(num_warmup_trials, num_trials,
                                               self._layer)
        (input_cast_time, half_precision_runtime, output_cast_time) = \
            self._profile(num_warmup_trials, num_trials,
                          half_precision_layer,
                          half_precision=True)
        total_half_precision_time = \
            input_cast_time + half_precision_runtime + output_cast_time
        speedup = full_precision_runtime / total_half_precision_time
        return (full_precision_runtime, input_cast_time,
                half_precision_runtime, output_cast_time)
        """
        print('%s: full=%f microseconds, '
              'half=%f + %f + %f = %f microseconds, '
              'speedup=%.2fx' % (str(self),
                                 full_precision_runtime,
                                 input_cast_time,
                                 half_precision_runtime,
                                 output_cast_time,
                                 total_half_precision_time,
                                 speedup))
        """

class LinearBenchmark(HalfPrecisionBenchmark):
    def __init__(self, n, in_features, out_features):
        self._n = n
        self._in_features = in_features
        self._out_features = out_features
        self._layer = torch.nn.Linear(in_features, out_features)
        self._layer.to('cuda')
        self._data = \
            torch.rand((n, in_features), dtype=torch.float32).to('cuda')

    @property
    def params(self):
        return '%d,%d,%d' % (self._n, self._in_features, self._out_features)

    def __str__(self):
        return ('%s (n=%d, in_features=%d, '
                'out_features=%d)') % ('Linear', self._n, self._in_features,
                                       self._out_features)


def get_linear_benchmarks(batch_size):
    linear_benchmarks = [
        LinearBenchmark(batch_size, 768, 2),
        LinearBenchmark(batch_size, 768, 768),
        LinearBenchmark(batch_size * 128, 768, 768),
        LinearBenchmark(batch_size * 128, 768, 3072),
        LinearBenchmark(batch_size * 128, 3072, 768),
    ]
    return linear_benchmarks

def main(args):
    if args.output_file is not None:
        output_file = open(args.output_file, 'w')
    else:
        output_file = None
    for batch_size in [1, 2, 4, 8, 16, 32, 64]:
        if output_file is None:
            print('batch_size=%d:' % (batch_size))
        for benchmark in get_linear_benchmarks(batch_size):
            results = benchmark.profile(1, 5)
            if output_file is not None:
                output_file.write('%d,%s,%f,%f,%f,%f\n' % (batch_size,
                                                           benchmark.params,
                                                           results[0],
                                                           results[1],
                                                           results[2],
                                                           results[3]))
            else:
                print('%s: full precision runtime = %f, '
                      'half precision runtime = %f + %f + %f = %f, '
                      'speedup = %.2fx' % (str(benchmark),
                                           results[0], results[1], results[2],
                                           results[3], sum(results[1:]),
                                           results[0] / sum(results[1:])))
        if output_file is None:
            print()

    if output_file is not None:
        output_file.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Benchmark 16-bit layers')
    parser.add_argument('--output_file', type=str, default=None)
    args = parser.parse_args()
    main(args)
