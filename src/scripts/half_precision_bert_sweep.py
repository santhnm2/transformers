import re
import subprocess

BASE_COMMAND = 'python bert_test_pytorch.py --eval --load_from_checkpoint --cache_dir cache'

print('Model,Batch Size,32-bit accuracy,16-bit accuracy,32-bit runtime,16-bit runtime')
for model in ['bert', 'distilbert']:
    checkpoint_dir = '%s_checkpoint'
    command = ('%s --model %s '
               '--checkpoint_dir %s_checkpoint') % (BASE_COMMAND, model, model)
    for batch_size in [1, 2, 4, 8, 16, 32, 64]:
        acc = []
        runtimes = []
        command = '%s --valid_batch_size %d' % (command, batch_size)
        for half_precision in [False, True]:
            if half_precision:
                final_command = '%s --half_precision' % (command)
            else:
                final_command = command
            output = \
                subprocess.check_output(final_command, shell=True).decode('utf-8').strip()
            match = re.search('Validation accuracy: (\d+\.\d+)%',
                              output.split('\n')[-3])
            acc.append(float(match.group(1)))
            #print(output.split('\n')[-2])
            match = re.search('Average per-batch runtime: (\d+\.\d+) seconds',
                              output.split('\n')[-2])
            runtimes.append(float(match.group(1)))
        print('%s,%d,%f,%f,%f,%f' % (model, batch_size, acc[0], acc[1],
                                     runtimes[0], runtimes[1]), flush=True)
