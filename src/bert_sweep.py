import subprocess
import re

BASE_COMMAND = 'python bert_experiments.py'

def experiment(model, task, epochs, linformer_k, lr, blocks, logdir):
    command = '%s --model %s --task %s --epochs %d --lr %f' % (BASE_COMMAND,
                                                               model, task,
                                                               epochs, lr)
    if blocks is not None:
        blocks_str = ','.join([str(block) for block in blocks])
        command = '%s --linformer_k %d --linformer_blocks %s' % (command,
                                                                 linformer_k,
                                                                 blocks_str)
    else:
        blocks_str = 'None'
    output = subprocess.check_output(command, shell=True).decode('utf-8').strip()
    logfile = os.path.join(logdir, 'blocks=%s.log' % (blocks_str))
    with open(logfile, 'w') as f:
        f.write(output)
    """
    iteration = 0
    x = []
    train_y = []
    valid_y = []
    for line in output.split('\n'):
        match = re.search(('Epoch (\d+), Iteration (\d+): '
                           'train_loss=([-+]?[0-9]*\.?[0-9]+), '
                           'valid_loss=([-+]?[0-9]*\.?[0-9]+), '
                           'valid_acc=([-+]?[0-9]*\.?[0-9]+)'), line)
        if match is not None:
            # epoch = int(match.group(1))
            iteration += int(match.group(2))
            train_loss = float(match.group(3))
            valid_loss = float(match.group(4))
            #valid_acc = float(match.group(5))
            x.append(iteration)
            train_y.append(train_loss)
            valid_y.append(valid_loss)
    """


def main():
    base_logdir = '/lfs/1/keshav2/logs/linformer'
    model = 'distilbert'
    task = 'mrpc'
    epochs = 1
    for k in [64]:
        k_logdir = os.path.join(base_logdir, 'k=%d' % (k))
        if not os.path.isdir(k_logdir):
            os.mkidr(k_logdir)
        for lr in [3e-5]:
            lr_logdir = os.path.join(k_logdir, 'lr=%f' % (lr))
            if not os.path.isdir(lr_logdir):
                os.mkdir(lr_logdir)
            for blocks in [None, [0], [1], [2], [3], [4], [5]]:
                if blocks is None:
                    blocks_str = 'None'
                else:
                    block_str = ','.join([str(block) for block in blocks])
                    experiment(model, task, epochs, k, lr, blocks, logdir)

if __name__=='__main__':
    main()
