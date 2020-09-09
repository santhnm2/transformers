import re
import subprocess

BASE_COMMAND = 'python bert_experiments.py'

def evaluate(model, task, max_seq_length):
    command = ('{0} --model {1} --task {2} --max_seq_length {3} '
               '--load_from_checkpoint --eval').format(
                    BASE_COMMAND, model, task, max_seq_length)
    output = subprocess.check_output(
                command, shell=True).decode('utf-8').strip()
    for line in output.split('\n'):
        match = \
            re.search('Validation accuracy: ([-+]?[0-9]*\.?[0-9]+)%', line)
        if match is not None:
            validation_accuracy = float(match.group(1))
            return validation_accuracy
    raise RuntimeError('Could not get accuracy!')

def train_with_linformer(model, task, max_seq_length, batch_size, epochs,
                         valid_acc_target, linformer_k, blocks):
    blocks_str = ','.join([str(block) for block in blocks])
    command = ('{0} --model {1} --task {2} --max_seq_length {3} '
               '--batch_size {4} --epochs {5} --valid_acc_target {6} '
               '--linformer_k {7} --linformer_blocks {8} '
               '--load_from_checkpoint').format(
                    BASE_COMMAND, model, task, max_seq_length, batch_size,
                    epochs, valid_acc_target, linformer_k, blocks_str)
    print(command)
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

    valid_acc_target = (evaluate(model, task, max_seq_length) - 1.0) / 100.0

    candidates = set(range(12))
    opt_sequence = []
    while len(candidates) > 0:
        best_candidate = None
        best_accuracy = 0.0
        found_new_opt = False

        for candidate in candidates:
            accuracy = train_with_linformer(model, task, max_seq_length,
                                            batch_size, epochs,
                                            valid_acc_target,
                                            linformer_k, [candidate])
            print('Candidate: {0}, accuracy={1: .2f}'.format(
                    candidate, accuracy))
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_candidate = candidate
                found_new_opt = True

        if not found_new_opt:
            break
        else:
            opt_sequence.append(candidate)
            candidates.remove(candidate)

        # TODO: Allow for optimization sequences > 1
        break

if __name__=='__main__':
    main()
