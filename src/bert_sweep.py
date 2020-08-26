import copy
#import multiprocessing
import os
import queue
import re
import subprocess
from multiprocessing.pool import ThreadPool

BASE_COMMAND = 'python bert_experiments.py'

def powerset(seq):
    """
    Returns all the subsets of this set. This is a generator.
    https://www.technomancy.org/python/powerset-generator-python/
    """
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]]+item
            yield item

def run_experiment(model, task, epochs, valid_acc_target,
                   linformer_k, lr, blocks, logdir, gpu_id):
    command = '%s --model %s --task %s --epochs %d --lr %f' % (BASE_COMMAND,
                                                               model, task,
                                                               epochs, lr)
    command = '%s --load_from_checkpoint --valid_acc_target %f' % (command, valid_acc_target)
    command = '%s --batch_size 64' % (command)
    if blocks is not None:
        blocks_str = ','.join([str(block) for block in blocks])
        command = '%s --linformer_k %d --linformer_blocks %s' % (command,
                                                                 linformer_k,
                                                                 blocks_str)
    else:
        blocks_str = 'None'
    env = copy.deepcopy(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print('===> Running \"%s\"...' % (command))
    output = subprocess.check_output(command, env=env, shell=True).decode('utf-8').strip()
    logfile = os.path.join(logdir, 'blocks=%s.log' % (blocks_str))
    with open(logfile, 'w') as f:
        f.write(output)


def main():
    base_logdir = '/lfs/1/keshav2/logs/linformer'
    epochs = 2
    num_gpus = 4
    pool = ThreadPool(num_gpus)

    valid_acc_targets = {
        'mrpc': 0.94,
        'sst-2': 0.9128,
    }

    experiments = queue.Queue()
    for model in ['distilbert']:
        model_logdir = os.path.join(base_logdir, 'model=%s' % (model))
        if not os.path.isdir(model_logdir):
            os.mkdir(model_logdir)
        for task in ['sst-2']:
            task_logdir = os.path.join(model_logdir, 'task=%s' % (task))
            if not os.path.isdir(task_logdir):
                os.mkdir(task_logdir)
            for target_delta in [0, 0.0025, 0.05]:
                valid_acc_target = valid_acc_targets[task] - target_delta
                valid_acc_target_logdir = \
                    os.path.join(task_logdir,
                                 'valid_acc_target=%.4f' % (valid_acc_target))
                if not os.path.isdir(valid_acc_target_logdir):
                    os.mkdir(valid_acc_target_logdir)
                for k in [128]:
                    k_logdir = \
                        os.path.join(valid_acc_target_logdir, 'k=%d' % (k))
                    if not os.path.isdir(k_logdir):
                        os.mkdir(k_logdir)
                    for lr in [3e-5]:
                        lr_logdir = os.path.join(k_logdir, 'lr=%f' % (lr))
                        if not os.path.isdir(lr_logdir):
                            os.mkdir(lr_logdir)
                        for blocks in list(powerset([0, 1, 2, 3, 4, 5])):
                            if len(blocks) == 0:
                                continue
                            experiments.put((model, task, epochs,
                                             valid_acc_target, k, lr, blocks,
                                             lr_logdir))
    gpu_queue = queue.Queue()
    for i in range(num_gpus):
        gpu_queue.put(i)
    while not experiments.empty():
        results = []
        while not gpu_queue.empty():
            experiment = experiments.get()
            gpu_id = gpu_queue.get()
            results.append(pool.apply_async(run_experiment, (*experiment, gpu_id)))
        results = [result.get() for result in results]
        for i in range(num_gpus):
            gpu_queue.put(i)


if __name__=='__main__':
    main()
