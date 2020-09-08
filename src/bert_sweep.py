import copy
#import multiprocessing
import os
import queue
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

BASE_COMMAND = 'python bert_experiments.py'

num_gpus = 4
gpu_queue = queue.Queue()
for i in range(num_gpus):
    gpu_queue.put(i)

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

def run_experiment(args):
    (model, task, epochs, batch_size, max_seq_length, valid_acc_target,
     linformer_k, lr, blocks, logdir, trial) = args
    gpu_id = gpu_queue.get()
    command = '%s --model %s --task %s --epochs %d --lr %f' % (BASE_COMMAND,
                                                               model, task,
                                                               epochs, lr)
    command = '%s --load_from_checkpoint --valid_acc_target %f' % (command, valid_acc_target)
    command = '%s --batch_size %d' % (command, batch_size)
    command = '%s --max_seq_length %d' % (command, max_seq_length)
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
    logfile = os.path.join(logdir, 'trial=%d.log' % (trial))
    with open(logfile, 'w') as f:
        f.write(output)
    gpu_queue.put(gpu_id)

def main():
    base_logdir = '/lfs/1/keshav2/logs/linformer'
    epochs = 2
    max_seq_length = 512
    batch_size = 16
    all_models = ['distilbert']
    all_tasks = ['mrpc']
    all_target_deltas = [0, 0.005, 0.01]
    all_k = [64, 128, 256]
    all_lr = [3e-5]
    num_trials = 5
    pool = ThreadPoolExecutor(max_workers=num_gpus)

    valid_acc_targets = {
        'mrpc': 0.8456,
        'sst-2': 0.9128,
    }

    experiments = []
    for model in all_models:
        model_logdir = os.path.join(base_logdir, 'model=%s' % (model))
        if not os.path.isdir(model_logdir):
            os.mkdir(model_logdir)
        for task in all_tasks:
            task_logdir = os.path.join(model_logdir, 'task=%s' % (task))
            if not os.path.isdir(task_logdir):
                os.mkdir(task_logdir)
            for target_delta in all_target_deltas:
                valid_acc_target = valid_acc_targets[task] - target_delta
                valid_acc_target_logdir = \
                    os.path.join(task_logdir,
                                 'valid_acc_target=%.4f' % (valid_acc_target))
                if not os.path.isdir(valid_acc_target_logdir):
                    os.mkdir(valid_acc_target_logdir)
                for k in all_k:
                    k_logdir = \
                        os.path.join(valid_acc_target_logdir, 'k=%d' % (k))
                    if not os.path.isdir(k_logdir):
                        os.mkdir(k_logdir)
                    for lr in all_lr:
                        lr_logdir = os.path.join(k_logdir, 'lr=%f' % (lr))
                        if not os.path.isdir(lr_logdir):
                            os.mkdir(lr_logdir)
                        for blocks in list(powerset([0, 1, 2, 3, 4, 5])):
                            if len(blocks) == 0:
                                continue
                            blocks_str = \
                                ','.join([str(block) for block in blocks])
                            blocks_logdir = \
                                os.path.join(lr_logdir, 'blocks=%s' % (blocks_str))
                            if not os.path.isdir(blocks_logdir):
                                os.mkdir(blocks_logdir)
                            for trial in range(num_trials):
                                experiments.append((model, task, epochs,
                                                    batch_size, max_seq_length,
                                                    valid_acc_target, k, lr,
                                                    blocks, blocks_logdir,
                                                    trial))

    pool.map(run_experiment, experiments)

if __name__=='__main__':
    main()
