from transformers import *
import torch

import argparse
import math
import time
import random

class HalfPrecisionLayer(torch.nn.Module):
    def __init__(self, name, layer):
        super(HalfPrecisionLayer, self).__init__()
        self._name = name + '_fp16'
        self._modules[self._name] = layer
        self._modules[self._name].half()

    def forward(self, *args, **kwargs):
        """
        print('HalfPrecisionLayer %s:' % (self._name))
        print('args:', args)
        print('kwargs:', kwargs)
        """
        half_precision_args = []
        half_precision_kwargs = {}
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.dtype == torch.float32:
                print('WARNING: cast!')
                half_precision_args.append(arg.half())
                assert(half_precision_args[-1].dtype == torch.float16)
            else:
                half_precision_args.append(arg)
        for key, val in kwargs.items():
            if isinstance(val, torch.Tensor) and arg.dtype == torch.float32:
                print('WARNING: cast!')
                half_precision_kwargs[key] = val.half()
                assert(half_precision_args[key].dtype == torch.float16)
            else:
                half_precision_kwargs[key] = val

        ret = self._modules[self._name].forward(*half_precision_args,
                                                **half_precision_kwargs)
        return ret
        """
        if ret is None:
            return ret
        elif not isinstance(ret, tuple):
            ret = (ret,)
        assert isinstance(ret, tuple)
        ret = list(ret)
        for i in range(len(ret)):
            if (isinstance(ret[i], torch.Tensor) and
                ret[i].dtype == torch.float16):
                ret[i] = ret[i].float()
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)
        """

def train(args, model, train_loader, valid_loader):
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, eps=1e-8)
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(input_ids=batch['input_ids'].to('cuda'),
                            attention_mask=batch['attention_mask'].to('cuda'))
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
                valid_acc = evaluate(args, model, valid_loader)
                print('===> Epoch %d, Iteration %d: '
                      'loss=%.3f, valid_acc=%.2f' % (epoch+1, i+1,
                                                     running_loss,
                                                     valid_acc),
                      end='\n' if done else '\r')
                running_loss = 0.0

def _evaluate(args, model, valid_loader):
    num_correct = 0.0
    with torch.no_grad():
        for batch in valid_loader:
            outputs = model(input_ids=batch['input_ids'].to('cuda'),
                            attention_mask=batch['attention_mask'].to('cuda'))
            logits = outputs[0]
            predictions = torch.argmax(logits, dim=1)
            num_correct += batch['labels'].to('cuda').eq(predictions).sum()
    return num_correct / len(valid_loader.dataset)

def evaluate(args, model, valid_loader, warm_up=False, profile=False):
    torch.cuda.empty_cache()
    if warm_up:
        _evaluate(args, model, valid_loader)
    if profile:
        start_time = time.time()
    valid_acc = _evaluate(args, model, valid_loader)
    if profile:
        runtime = time.time() - start_time
        return valid_acc, runtime
    return valid_acc

def cuda_profile(args, model, valid_loader):
    batch = next(iter(valid_loader))
    input_ids = batch['input_ids'].to('cuda')
    attention_mask = batch['attention_mask'].to('cuda')
    torch.cuda.synchronize()
    #with torch.autograd.profiler.profile(use_cuda=True, record_shapes=True) as prof:
    with torch.cuda.profiler.profile():
        with torch.no_grad():
            # Warm-up.
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)
            with torch.autograd.profiler.emit_nvtx():
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask)
    #print(prof)
    import pdb
    pdb.set_trace()

def cast_to_half_precision(model, verbose=False):
    stack = [(None, model, 'model', 1)]
    while len(stack) > 0:
        (parent, module, name, depth) = stack.pop(-1)
        indent = ''
        for i in range(depth-1):
            indent += ' '
        if verbose:
            print('%s%s (%s)' % (indent, name, module._get_name()))
        children = list(module.named_children())
        is_leaf = len(children) == 0
        if is_leaf:
            if verbose:
                print('%sReplacing %s (%s) '
                      'with HalfPrecisionLayer' % (indent, name,
                                                   module._get_name()))
            parent._modules[name] = HalfPrecisionLayer(name, module)
            if verbose:
                print('%s%s[%s] = %s' % (indent, parent._get_name(), name,
                                         parent._modules[name]._get_name()))
        else:
            for i, (child_name, child) in enumerate(children[::-1]):
                stack.append((module, child, child_name, depth+1))

def skip_encoder_blocks(args, model):
    if args.model == 'bert':
        num_layers = len(model.bert.encoder.layer)
    elif args.model == 'distilbert':
        num_layers = len(model.distilbert.transformer.layer)
    if max(args.encoder_blocks_to_skip) >= num_layers:
        raise ValueError('Skipping invalid encoder blocks!')

    for block in sorted(args.encoder_blocks_to_skip, reverse=True):
        if args.model == 'bert':
            del model.bert.encoder.layer[block]
        elif args.model == 'distilbert':
            del model.distilbert.transformer.layer[block]

def depth_first_traversal(model):
    stack = [(None, None, model, 'model', 1)]
    while len(stack) > 0:
        (parent, index, module, name, depth) = stack.pop(-1)
        indent = ''
        for i in range(depth-1):
            indent += ' '
        print('%s%s (%s)' % (indent, name, module._get_name()))
        children = list(module.named_children())
        is_leaf = len(children) == 0
        if is_leaf:
            pass
        else:
            for i, (name, child) in enumerate(children[::-1]):
                stack.append((module, len(children) - 1 - i, child, name,
                              depth+1))

def main(args):
    if args.model == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased',
                                                  cache_dir=args.cache_dir)
    elif args.model == 'distilbert':
        tokenizer = \
            DistilBertTokenizer.from_pretrained('distilbert-base-cased',
                                                cache_dir=args.cache_dir)
    if args.load_from_checkpoint:
        if args.checkpoint_dir is None:
            raise ValueError('No checkpoint dir specified!')
        print('===> Loading from checkpoint...')
        if args.model == 'bert':
            model = BertForSequenceClassification.from_pretrained(args.checkpoint_dir)
        elif args.model == 'distilbert':
            model = DistilBertForSequenceClassification.from_pretrained(args.checkpoint_dir)
    else:
        if args.model == 'bert':
            model = \
                BertForSequenceClassification.from_pretrained('bert-base-cased',
                                                              cache_dir=args.cache_dir)
        elif args.model == 'distilbert':
            model = \
                DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased',
                                                                    cache_dir=args.cache_dir)
    model.to('cuda')

    if args.half_precision:
        cast_to_half_precision(model)

    if args.encoder_blocks_to_skip is not None:
        skip_encoder_blocks(args, model)

    glue_data_training_args = \
        GlueDataTrainingArguments(task_name='MRPC',
                                  data_dir=args.data_dir)
    train_dataset = GlueDataset(args=glue_data_training_args,
                                tokenizer=tokenizer,
                                mode='train',
                                cache_dir='cache')
    valid_dataset = GlueDataset(args=glue_data_training_args,
                                tokenizer=tokenizer,
                                mode='dev',
                                cache_dir='cache')
    collator = DefaultDataCollator()
    train_loader = \
        torch.utils.data.DataLoader(train_dataset,
                                    batch_size=args.batch_size,
                                    collate_fn=collator.collate_batch)
    valid_loader = \
        torch.utils.data.DataLoader(valid_dataset,
                                    batch_size=args.valid_batch_size,
                                    collate_fn=collator.collate_batch)
    num_validation_steps = \
        math.ceil(len(valid_dataset) / args.valid_batch_size)

    if not args.eval:
        train(args, model, train_loader, valid_loader)
        if args.save_to_checkpoint:
            if args.checkpoint_dir is None:
                raise ValueError('No checkpoint dir specified!')
            print('===> Saving to checkpoint...')
            model.save_pretrained(args.checkpoint_dir)

    cuda_profile(args, model, valid_loader)
    """
    valid_acc, runtime = evaluate(args, model, valid_loader, warm_up=True,
                                 profile=True)
    print('===> Validation accuracy: %.2f%%' % (valid_acc * 100.0))
    print('===> Runtime: %.3f seconds' % (runtime))
    """

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Test HuggingFace (Distil)BERT for sequence classification')
    parser.add_argument('--model', choices=['bert', 'distilbert'],
                        required=True, help='Model to run')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Cache directory')
    parser.add_argument('--load_from_checkpoint', action='store_true',
                        default=False,
                        help='If set, load model from checkpoint')
    parser.add_argument('--save_to_checkpoint', action='store_true',
                        default=False,
                        help='If set, save model to checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Checkpoint directory')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='If set, only run evaluation on validation set')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs to train for')
    parser.add_argument('--encoder_blocks_to_skip', type=int, nargs='+',
                        default=None,
                        help='Indices of encoder layers to skip')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--valid_batch_size', type=int, default=64,
                        help='Validation batch size')
    parser.add_argument('--num_warmup_evals', type=int, default=0,
                        help='Number of warmup validation rounds to run')
    parser.add_argument('--data_dir', type=str, default='glue_data/MRPC',
                        help='MRPC data dir')
    parser.add_argument('--print_freq', type=int, default=100,
                        help='Print frequency for training')
    parser.add_argument('--half_precision', action='store_true', default=False,
                        help='If set, cast all layers to half precision')
    args = parser.parse_args()
    main(args)
