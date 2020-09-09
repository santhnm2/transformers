from transformers import *
import torch

import argparse
import math
import numpy as np
import os
import random
import time

def train(args, model, train_loader, valid_loader, linformer=None):
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8)
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            outputs = model(input_ids=batch['input_ids'].to('cuda'),
                            attention_mask=batch['attention_mask'].to('cuda'),
                            linformer=linformer)
            logits = outputs[0]
            one_hot_labels = \
                torch.cuda.FloatTensor(batch['labels'].size(0), 2).zero_()
            one_hot_labels.scatter_(1, batch['labels'].to('cuda').unsqueeze(1), 1)
            loss = criterion(logits, one_hot_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()
            done = i == len(train_loader) - 1
            if done or (i+1) % args.print_freq == 0:
                running_loss /= args.print_freq
                valid_acc, valid_loss = evaluate(args, model, valid_loader,
                                                 linformer=linformer)
                print('===> Epoch %d, Iteration %d: '
                      'train_loss=%.4f, valid_loss=%.4f, '
                      'valid_acc=%.4f' % (epoch+1, i+1, running_loss,
                                          valid_loss, valid_acc))
                running_loss = 0.0
                if (args.valid_acc_target is not None and
                    valid_acc >= args.valid_acc_target):
                    return

def _evaluate(args, model, valid_loader, linformer=None):
    all_correct = []
    per_batch_runtimes = []
    criterion = torch.nn.BCEWithLogitsLoss()
    running_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            start_time = time.time()
            outputs = model(input_ids=batch['input_ids'].to('cuda'),
                            attention_mask=batch['attention_mask'].to('cuda'),
                            linformer=linformer)
            torch.cuda.synchronize()
            per_batch_runtimes.append(time.time() - start_time)
            logits = outputs[0]
            one_hot_labels = \
                torch.cuda.FloatTensor(batch['labels'].size(0), 2).zero_()
            one_hot_labels.scatter_(1, batch['labels'].to('cuda').unsqueeze(1), 1)
            loss = criterion(logits, one_hot_labels)
            running_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct = batch['labels'].to('cuda').eq(predictions).nonzero()
            correct += (valid_loader.batch_size * i)
            all_correct += correct.flatten().tolist()
    acc = float(len(all_correct)) / len(valid_loader.dataset)
    loss = running_loss / i
    return acc, loss, all_correct, per_batch_runtimes

def evaluate(args, model, valid_loader, warm_up=False, profile=False,
             linformer=None):
    model.eval()
    if profile:
        torch.cuda.empty_cache()
    if warm_up:
        _evaluate(args, model, valid_loader, linformer=linformer)
    valid_acc, valid_loss, all_correct, per_batch_runtimes = \
        _evaluate(args, model, valid_loader, linformer=linformer)
    if profile:
        return valid_acc, valid_loss, all_correct, per_batch_runtimes
    return valid_acc, valid_loss

def cuda_profile(args, model, valid_loader):
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    batch = next(iter(valid_loader))
    input_ids = batch['input_ids'].to('cuda')
    attention_mask = batch['attention_mask'].to('cuda')
    with torch.no_grad():
        # Warm-up.
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask)
        torch.cuda.synchronize()
        with torch.autograd.profiler.profile(use_cuda=True, record_shapes=True) as prof:
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)
    print(prof)

def get_linformer(args, model, seq_len):
    """Generates the projection matrices for Linformer."""
    linformer = {}
    if args.linformer_blocks == 'all':
        linformer_blocks = \
            set(list(range(6 if args.model == 'distilbert' else 12)))
    else:
        linformer_blocks = \
            set([int(x) for x in args.linformer_blocks.split(',')])
    if args.linformer_k is not None:
        if args.share_linformer:
            e = torch.normal(mean=0,
                             std=(1.0 / np.sqrt(args.linformer_k)),
                             size=(seq_len, args.linformer_k)).cuda()
            f = torch.normal(mean=0,
                             std=(1.0 / np.sqrt(args.linformer_k)),
                             size=(seq_len, args.linformer_k)).cuda()
        for i in range(6 if args.model == 'distilbert' else 12):
            if i not in linformer_blocks:
                linformer[i] = None
                continue
            linformer[i] = {}
            if args.share_linformer:
                linformer[i]['e'] = e
                linformer[i]['f'] = f
            else:
                linformer[i]['e'] = \
                    torch.normal(mean=0,
                                 std=(1.0 / np.sqrt(args.linformer_k)),
                                 size=(seq_len, args.linformer_k)).cuda()
                linformer[i]['f'] = \
                    torch.normal(mean=0,
                                 std=(1.0 / np.sqrt(args.linformer_k)),
                                 size=(seq_len, args.linformer_k)).cuda()
    else:
        for i in range(6 if args.model == 'distilbert' else 12):
            linformer[i] = None
    return linformer

def main(args):
    if args.checkpoint_dir is not None:
        if not os.path.isdir(args.checkpoint_dir):
            raise ValueError(
                'Invalid checkpoint dir "%s"' % (args.checkpoint_dir))
        checkpoint_dir = os.path.join(args.checkpoint_dir,
                                      'model=%s_task=%s' % (args.model,
                                                            args.task))
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
    if args.model == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased',
                                                  cache_dir=args.cache_dir,
                                                  max_length=args.max_seq_length,
                                                  truncation=True,
                                                  padding=True)
    elif args.model == 'distilbert':
        tokenizer = \
            DistilBertTokenizer.from_pretrained('distilbert-base-cased',
                                                cache_dir=args.cache_dir,
                                                max_length=args.max_seq_length,
                                                truncation=True,
                                                padding=True)
    elif args.model == 'roberta':
        tokenizer = \
            RobertaTokenizer.from_pretrained('roberta-base',
                                             cache_dir=args.cache_dir,
                                             max_length=args.max_seq_length,
                                             truncation=True,
                                             padding=True)
    if args.load_from_checkpoint:
        if args.checkpoint_dir is None:
            raise ValueError('No checkpoint dir specified!')
        print('===> Loading from checkpoint...')
        if args.model == 'bert':
            model = BertForSequenceClassification.from_pretrained(checkpoint_dir)
        elif args.model == 'distilbert':
            model = DistilBertForSequenceClassification.from_pretrained(checkpoint_dir)
        elif args.model == 'roberta':
            model = RobertaForSequenceClassification.from_pretrained(checkpoint_dir)
    else:
        if args.model == 'bert':
            model = BertForSequenceClassification.from_pretrained(
                    'bert-base-cased', cache_dir=args.cache_dir)
        elif args.model == 'distilbert':
            model = DistilBertForSequenceClassification.from_pretrained(
                    'distilbert-base-cased', cache_dir=args.cache_dir)
        elif args.model == 'roberta':
            model = RobertaForSequenceClassification.from_pretrained(
                    'roberta-base', cache_dir=args.cache_dir)
    model.to('cuda')

    if args.task == 'mrpc':
        task_name = 'MRPC'
    elif args.task == 'sst-2':
        task_name = 'SST-2'
    elif args.task == 'qnli':
        task_name = 'QNLI'
    elif args.task == 'qqp':
        task_name = 'QQP'
    else:
        raise NotImplementedError(args.task)
    data_dir = os.path.join(args.data_dir, task_name)
    glue_data_training_args = \
        GlueDataTrainingArguments(task_name=task_name,
                                  data_dir=data_dir,
                                  max_seq_length=args.max_seq_length)
    train_dataset = GlueDataset(args=glue_data_training_args,
                                tokenizer=tokenizer,
                                mode='train',
                                cache_dir=args.cache_dir)
    valid_dataset = GlueDataset(args=glue_data_training_args,
                                tokenizer=tokenizer,
                                mode='dev',
                                cache_dir=args.cache_dir)
    train_loader = \
        torch.utils.data.DataLoader(train_dataset,
                                    batch_size=args.batch_size,
                                    collate_fn=default_data_collator)
    valid_loader = \
        torch.utils.data.DataLoader(valid_dataset,
                                    batch_size=args.valid_batch_size,
                                    collate_fn=default_data_collator)
    num_validation_steps = \
        math.ceil(len(valid_dataset) / args.valid_batch_size)

    linformer = get_linformer(args, model, args.max_seq_length)

    if not args.eval:
        train(args, model, train_loader, valid_loader, linformer=linformer)
        if args.save_to_checkpoint:
            if args.save_checkpoint_dir is not None:
                save_checkpoint_dir = args.save_checkpoint_dir
                if not os.path.isdir(save_checkpoint_dir):
                    os.mkdir(checkpoint_dir)
            else:
                save_checkpoint_dir = checkpoint_dir
            print('===> Saving to checkpoint...')
            model.save_pretrained(save_checkpoint_dir)

    valid_acc, valid_loss, all_correct, per_batch_runtimes = \
        evaluate(args, model, valid_loader, warm_up=True, profile=True,
                 linformer=linformer)
    if args.print_correct:
        incorrect = sorted(set(range(len(valid_dataset))) - set(all_correct))
        print('===> Correct examples:', ' '.join([str(x) for x in all_correct]))
        print('===> Incorrect examples:',
              ' '.join([str(x) for x in incorrect]))
    print('===> Validation loss: %.3f' % (valid_loss))
    print('===> Validation accuracy: %.2f%%' % (valid_acc * 100.0))
    print('===> Median per-batch runtime: '
          '%.5f ms' % (1000 * np.median(per_batch_runtimes)))
    print('===> Total runtime: %.5f seconds' % (sum(per_batch_runtimes)))

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Approximate HuggingFace (Distil)Bert')
    parser.add_argument('--model', choices=['bert', 'distilbert', 'roberta'],
                        required=True, help='Model to run')
    parser.add_argument('--task', required=True, type=str,
                        choices=['cola', 'mrpc', 'sst-2', 'qqp', 'qnli', 'rte',
                                 'wnli'],
                        help='GLUE task')
    parser.add_argument('--cache_dir', type=str, default='/lfs/1/keshav2/.cache/',
                        help='Cache directory')
    parser.add_argument('--load_from_checkpoint', action='store_true',
                        default=False,
                        help='If set, load model from checkpoint')
    parser.add_argument('--save_to_checkpoint', action='store_true',
                        default=False,
                        help='If set, save model to checkpoint')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='/lfs/1/keshav2/bert_checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='If set, only run evaluation on validation set')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--valid_batch_size', type=int, default=64,
                        help='Validation batch size')
    parser.add_argument('--num_warmup_evals', type=int, default=0,
                        help='Number of warmup validation rounds to run')
    parser.add_argument('--data_dir', type=str, default='/lfs/1/keshav2/data/glue/',
                        help='GLUE data dir')
    parser.add_argument('--print_freq', type=int, default=5,
                        help='Print frequency for training')
    parser.add_argument('--linformer_k', type=int, default=None,
                        help='Linformer k parameter')
    parser.add_argument('--share_linformer', action='store_true',
                        default=False,
                        help=('If set, share Linfomrer projection matrices '
                              'across attention blocks'))
    parser.add_argument('--linformer_blocks', type=str, default='all',
                        help='List of attention blocks to apply Linformer to')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--print_correct', action='store_true',
                        default=False, help='If set, print correct examples')
    parser.add_argument('--max_seq_length', type=int, default=128,
                        help='Maximum sequence length')
    parser.add_argument('--valid_acc_target', type=float, default=None,
                        help='If set, break if validation accuracy hits target')
    parser.add_argument('--save_checkpoint_dir', type=str, default=None,
                        help='Directory to save checkpoint to')
    args = parser.parse_args()
    main(args)
