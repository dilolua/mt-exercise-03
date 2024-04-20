# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.onnx
import torch.nn as nn
from collections import defaultdict
import data
import model


parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--data', type=str, default='./data//data/guthenberg',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2, nargs="+",
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='use CUDA')
parser.add_argument('--mps', action='store_true', default=False,
                    help='enables macOS GPU training')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true',
                    help='do a dry run without training the model')
parser.add_argument('--ppl-log', action='store_true',
                    help='log the perplexities of each model after each epoch')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda.")
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    if not args.mps:
        print("WARNING: You have mps device, to enable macOS GPU run with --mps.")

use_mps = args.mps and torch.backends.mps.is_available()
if args.cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)


# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
# initialize model list
models_drpout = {}

# for all different dropout values, initialize a model
for dropout in args.dropout:
    # print(args.model, ntokens, args.emsize, args.nhid, args.nlayers, dropout, args.tied)
    if args.model == 'Transformer':
        mymodel = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, dropout).to(device)
    else:
        mymodel = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, dropout, args.tied).to(
            device)
    models_drpout[dropout] = mymodel
# print(models_drpout)
criterion = nn.NLLLoss()


###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target


def evaluate(data_source, any_model):
    # Turn on evaluation mode which disables dropout.
    any_model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = any_model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            if args.model == 'Transformer':
                output = any_model(data)
                output = output.view(-1, ntokens)
            else:
                output, hidden = any_model(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


training_log = defaultdict(list)



def train(model):
    # Turn on training mode which enables dropout.

    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        if args.model == 'Transformer':
            output = model(data)
            output = output.view(-1, ntokens)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch,
                len(train_data) // args.bptt,
                lr,
                elapsed * 1000 / args.log_interval,
                cur_loss,
                math.exp(cur_loss)))

            if args.ppl_log:
                # training_log[modl.dropoutvalue].append((epoch, math.exp(cur_loss)))
                training_log[model.dropoutvalue].append(math.exp(cur_loss))

            total_loss = 0
            start_time = time.time()

        if args.dry_run:
            break


def export_onnx(path, batch_size, seq_len, any_model):
    print('The model is also exported in ONNX format at {}.'.format(os.path.realpath(args.onnx_export)))
    any_model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = mymodel.init_hidden(batch_size)
    torch.onnx.export(any_model, (dummy_input, hidden), path)

    # Loop over epochs.


# lr = args.lr
# best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
validation_log = {}
# STORE MODEL NAMES FOR LATER RETRIEVAL
saved_models_str = {}
try:
    for drpt, mdl in models_drpout.items():
        # print(f'drpt{drpt}')
        lr = args.lr
        best_val_loss = None

        if args.ppl_log:
            validation_log[drpt] = []
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train(mdl)
            val_loss = evaluate(val_data, mdl)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             val_loss, math.exp(val_loss)))
            print('-' * 89)

            # saving the current epochs perplexity of the model into the log-dict IF the logging is turned on
            if args.ppl_log:
                validation_log[drpt].append(math.exp(val_loss))
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                if args.save.endswith('.pt'):
                    name = f'{args.save[:-3]}_drpt{drpt}{args.save[-3:]}'
                else:
                    name = f'{args.save}_drpt{drpt}.pt'
                saved_models_str[drpt] = name
                # print(name)

                with open(name, 'wb') as f:
                    torch.save(mdl, f)
                best_val_loss = val_loss
            else:
                lr /= 4.0

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
    print(f'log: {validation_log}')

test_log = {}
# for drpt, mdl in models_drpout.items():

# print(f'modelnames: {saved_models_str}')

for modelname in saved_models_str.values():
    # Load the best saved model.
    # print(f'modelname: {modelname}')
    with open(modelname, 'rb') as f:
        modl = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        # Currently, only rnn model supports flatten_parameters function.
        if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
            modl.rnn.flatten_parameters()

    # Run on test data.
    test_loss = evaluate(test_data, modl)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

    test_log[modl.dropoutvalue] = [math.exp(test_loss)]

    if len(args.onnx_export) > 0:
        # Export the model in ONNX format.
        export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt, modl=modl)

tofiles = (('training.csv', training_log), ('validation.csv', validation_log), ('test.csv', test_log))


def log_file(logtuple: tuple):
    name, logdict = logtuple
    line1 = name
    with open(f'{name}', 'w', encoding='utf-8') as logfile:
        for dropout in logdict.keys():
            line1 = f'{line1},{dropout}'
        logfile.write(line1)
        if name == 'test.csv':
            x = 1
        else:
            x = args.epochs
        for i in range(x):
            line = f'\n{i + 1}'
            for dropout in logdict.keys():
                # print(name, dropout)
                line = f'{line},{logdict[dropout][i]}'
            logfile.write(line)


if args.dry_run is False:
    for tup in tofiles:
        log_file(tup)

# Function to log the performance metrics to a CSV file
def log_metrics_to_csv(logs, file_name):
    with open(file_name, 'w') as f:
        for key in logs.keys():
            f.write("%s,%s\n" % (key, logs[key]))
    print(f"Performance metrics logged to {file_name}")

# Log the performance metrics to a CSV file
log_metrics_to_csv(test_log, 'performance_metrics.csv')


