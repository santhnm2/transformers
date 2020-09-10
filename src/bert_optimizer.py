import os
import re
import subprocess

BASE_COMMAND = 'python bert_experiments.py'

def evaluate(model, task, max_seq_length, checkpoint_dir,
             linformer_k=None, blocks=None):
    command = ('{0} --model {1} --task {2} --max_seq_length {3} '
               '--load_from_checkpoint --checkpoint_dir {4} --eval').format(
                    BASE_COMMAND, model, task, max_seq_length, checkpoint_dir)
    if linformer_k is not None:
        assert(blocks is not None)
        blocks_str = ','.join([str(block) for block in blocks])
        command = '{0} --linformer_k {1} --linformer_blocks {2}'.format(
                    command, linformer_k, blocks_str)
    output = subprocess.check_output(
                command, shell=True).decode('utf-8').strip()
    for line in output.split('\n'):
        match = re.search('Validation loss: ([-+]?[0-9]*\.?[0-9]+)', line)
        if match is not None:
            validation_loss = float(match.group(1))
            continue
        match = re.search('Validation accuracy: ([-+]?[0-9]*\.?[0-9]+)%', line)
        if match is not None:
            validation_accuracy = float(match.group(1))
            continue
        match = re.search(
                'Median per-batch runtime: ([-+]?[0-9]*\.?[0-9]+) ms', line)
        if match is not None:
            runtime = float(match.group(1))
            return validation_loss, validation_accuracy, runtime
    raise RuntimeError('Could not get accuracy and loss!')

def train(model, task, max_seq_length, batch_size, epochs,
          valid_acc_target, max_valid_loss, linformer_k, blocks,
          input_checkpoint_dir, output_checkpoint_dir):
    blocks_str = ','.join([str(block) for block in blocks])
    command = ('{0} --model {1} --task {2} --max_seq_length {3} '
               '--batch_size {4} --epochs {5} --valid_acc_target {6} '
               '--max_valid_loss {7} --linformer_k {8} --linformer_blocks {9} '
               '--load_from_checkpoint --checkpoint_dir {10} '
               '--save_to_checkpoint --save_checkpoint_dir {11}').format(
                    BASE_COMMAND, model, task, max_seq_length, batch_size,
                    epochs, valid_acc_target, max_valid_loss, linformer_k,
                    blocks_str, input_checkpoint_dir, output_checkpoint_dir)
    output = subprocess.check_output(
                command, shell=True).decode('utf-8').strip()
    for line in output.split('\n'):
        match = \
            re.search('Validation accuracy: ([-+]?[0-9]*\.?[0-9]+)%', line)
        if match is not None:
            validation_accuracy = float(match.group(1))
            return validation_accuracy
    raise RuntimeError('Could not get accuracy!')

def main():
    model = 'roberta'
    task = 'mrpc'
    batch_size = 16
    epochs = 1
    max_seq_length = 512
    linformer_k = 128
    base_checkpoint_dir = '/lfs/1/keshav2/bert_checkpoints'

    valid_loss, valid_acc, orig_runtime = evaluate(model, task, max_seq_length,
                                                   base_checkpoint_dir)
    valid_acc_target = valid_acc - 1.0
    max_valid_loss = valid_loss * 1.75

    candidates = set(range(12))
    opt_sequence = []
    current_checkpoint_dir = base_checkpoint_dir
    round_num = 0
    best_accuracy = 0.0
    print('Validation accuracy target: {0:.2f}%%'.format(valid_acc_target))
    print('Maximum validation loss: {0:.4f}'.format(max_valid_loss))
    print('Original runtime: {0:.2f} ms\n'.format(orig_runtime))
    while len(candidates) > 0:
        best_candidate = None
        found_new_opt = False

        print('*** Round {0} ***'.format(round_num))
        for candidate in candidates:
            blocks = sorted(opt_sequence + [candidate])
            blocks_str = ','.join([str(block) for block in blocks])
            output_checkpoint_dir = \
                os.path.join(base_checkpoint_dir,
                             'optimizer_blocks={0}'.format(blocks_str))
            if not os.path.isdir(output_checkpoint_dir):
                os.mkdir(output_checkpoint_dir)
            accuracy = train(model, task, max_seq_length, batch_size, epochs,
                             valid_acc_target / 100.0, max_valid_loss,
                             linformer_k, blocks, current_checkpoint_dir,
                             output_checkpoint_dir)
            print('Candidate: {0}, accuracy={1:.2f}'.format(
                    candidate, accuracy))
            if accuracy >= valid_acc_target:
                if best_candidate is None or accuracy >= best_accuracy:
                    best_accuracy = accuracy
                    best_candidate = candidate
                    found_new_opt = True

        print()
        if not found_new_opt:
            break
        else:
            round_num += 1
            opt_sequence.append(best_candidate)
            candidates.remove(best_candidate)
            blocks = sorted(opt_sequence)
            blocks_str = ','.join([str(block) for block in blocks])
            current_checkpoint_dir = \
                os.path.join(base_checkpoint_dir,
                             'optimizer_blocks={0}'.format(blocks_str))

    if len(opt_sequence) == 0:
        print('Could not optimize model!')
        return

    blocks = sorted(opt_sequence)
    blocks_str = ','.join([str(block) for block in blocks])
    output_checkpoint_dir = \
        os.path.join(base_checkpoint_dir,
                     'optimizer_blocks={0}'.format(blocks_str))
    opt_loss, opt_acc, opt_runtime = evaluate(model, task, max_seq_length,
                                             output_checkpoint_dir,
                                             linformer_k, blocks)

    print('Final optimized sequence: {0}'.format(opt_sequence))
    print('Final optimized accuracy: {0:.2f}%'.format(opt_acc))
    print('Final optimized runtime: {0:.2f} ms'.format(opt_runtime))
    print('Speedup: {0:.2f}x'.format(orig_runtime / opt_runtime))

if __name__=='__main__':
    main()
