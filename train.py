''' Sentence VAE '''
import os
import sys
import json
import time
import math
import random
import argparse
import ipdb as pdb
import logging as log

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
import numpy as np
from multiprocessing import cpu_count
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict

from ptb import PTB
from utils import to_var, idx2word, experiment_name
from model import SentenceVAE, SentenceAE

SCR_PREFIX = '/misc/vlgscratch4/BowmanGroup/awang/'
EPS = 1e-3

def kl_anneal_function(anneal_function, step, k, x0):
    if anneal_function == 'logistic':
        return float(1/(1+np.exp(-k*(step-x0))))
    elif anneal_function == 'linear':
        return min(1, step/x0)

def loss_fn(NLL, logp, target, length, mean, logv, anneal_function, step, k, x0):

    # cut-off unnecessary padding from target, and flatten
    target = target[:, :torch.max(length).data[0]].contiguous().view(-1)
    logp = logp.view(-1, logp.size(2))

    # Negative Log Likelihood
    nll_loss = NLL(logp, target)

    # KL Divergence
    kl_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
    kl_weight = kl_anneal_function(anneal_function, step, k, x0)

    return nll_loss, kl_loss, kl_weight

def main(arguments):
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', help='random seed', type=int, default=19)
    parser.add_argument('--run_dir', help='prefix to save ckpts to', type=str,
                        default=SCR_PREFIX + 'ckpts/svae/test/')
    parser.add_argument('--log_file', help='file to log to', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('--max_sequence_length', type=int, default=40)
    parser.add_argument('--min_occ', type=int, default=1)
    parser.add_argument('--max_vocab_size', type=int, default=30000)
    parser.add_argument('--test', action='store_true')

    parser.add_argument('-ep', '--epochs', type=int, default=10)
    parser.add_argument('-bs', '--batch_size', type=int, default=128)
    parser.add_argument('-o', '--optimizer', type=str, choices=['sgd', 'adam'], default='adam')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('--lr_decay_factor', type=float, default=0.5)
    parser.add_argument('-p', '--patience', type=int, default=5)
    parser.add_argument('--sched_patience', type=int, default=0)
    parser.add_argument('-mg', '--max_grad_norm', type=float, default=5.)

    parser.add_argument('-m', '--model', type=str, choices=['vae', 'ae'], default='vae')
    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, choices=['rnn', 'lstm', 'gru'],
                        default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=512)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')
    parser.add_argument('-ls', '--latent_size', type=int, default=16)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0.5)

    parser.add_argument('-d', '--denoise', action='store_true')
    parser.add_argument('-pd', '--prob_drop', type=float, default=0.1)
    parser.add_argument('-ps', '--prob_swap', type=float, default=0.1)

    parser.add_argument('-af', '--anneal_function', type=str, choices=['logistic', 'linear'],
                        default='logistic')
    parser.add_argument('-k', '--k', type=float, default=0.0025)
    parser.add_argument('-x0', '--x0', type=int, default=2500)

    parser.add_argument('-v', '--print_every', type=int, default=50)
    parser.add_argument('-tb', '--tensorboard_logging', action='store_true')

    args = parser.parse_args(arguments)

    log.basicConfig(format="%(asctime)s: %(message)s", level=log.INFO, datefmt='%m/%d %I:%M:%S %p')
    if args.log_file:
        log.getLogger().addHandler(log.FileHandler(args.log_file))
    log.info(args)
    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())

    seed = random.randint(1, 10000) if args.seed < 0 else args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    splits = ['train', 'valid'] + (['test'] if args.test else [])
    datasets = OrderedDict()
    for split in splits:
        datasets[split] = PTB(
            data_dir=args.data_dir,
            split=split,
            create_data=args.create_data,
            max_sequence_length=args.max_sequence_length,
            min_occ=args.min_occ)

    if args.model == 'vae':
        model = SentenceVAE(args, datasets['train'].get_w2i(),
                            embedding_size=args.embedding_size,
                            rnn_type=args.rnn_type,
                            hidden_size=args.hidden_size,
                            word_dropout=args.word_dropout,
                            latent_size=args.latent_size,
                            num_layers=args.num_layers,
                            bidirectional=args.bidirectional)
    elif args.model == 'ae':
        model = SentenceAE(args, datasets['train'].get_w2i(),
                           embedding_size=args.embedding_size,
                           rnn_type=args.rnn_type,
                           hidden_size=args.hidden_size,
                           word_dropout=args.word_dropout,
                           latent_size=args.latent_size,
                           num_layers=args.num_layers,
                           bidirectional=args.bidirectional)
    if args.denoise:
        log.info("DENOISING!")
    if torch.cuda.is_available():
        model = model.cuda()
    log.info(model)

    if args.tensorboard_logging:
        writer = SummaryWriter(os.path.join(args.run_dir, experiment_name(args, ts)))
        writer.add_text("model", str(model))
        writer.add_text("args", str(args))
        writer.add_text("ts", ts)
    save_model_path = args.run_dir
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    NLL = torch.nn.NLLLoss(size_average=False, ignore_index=datasets['train'].pad_idx)
    params = model.parameters()
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(params, lr=args.learning_rate)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(params, lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=args.lr_decay_factor,
                                                     patience=args.sched_patience,
                                                     verbose=True)
    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    batch_size = args.batch_size
    step, stop_training = 0, 0
    global_tracker = {'best_epoch': -1, 'best_score': -1, 'history': []}
    for epoch in range(args.epochs):
        if stop_training:
            break
        for split in splits:
            tracker = defaultdict(tensor)
            exs = [ex for ex in datasets[split].data.values()]
            random.shuffle(exs)
            n_batches = math.ceil(len(exs) / batch_size)

            # Enable/Disable Dropout
            if split == 'train':
                log.info("***** Epoch %02d *****", epoch)
                log.info("Training...")
                model.train()
            else:
                log.info("Validating...")
                model.eval()

            #for iteration, batch in enumerate(data_loader):
            for iteration in range(n_batches):
                raw_batch = exs[iteration*batch_size:(iteration+1)*batch_size]
                batch = model.prepare_batch([e['input'] for e in raw_batch])
                batch['src_length'] = model.tensor(batch['src_length']).long()
                batch['trg_length'] = model.tensor(batch['trg_length']).long()

                b_size = batch['input'].size(0)
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = to_var(v)

                # Forward pass
                logp, mean, logv, z = model(batch['input'], batch['target'],
                                            batch['src_length'], batch['trg_length'])

                # loss calculation
                nll_loss, kl_loss, kl_weight = model.loss_fn(logp, batch['target'],
                                                       batch['trg_length'], mean, logv,
                                                       args.anneal_function, step,
                                                       args.k, args.x0)
                loss = (nll_loss + kl_weight * kl_loss) / b_size
                nll_loss /= b_size
                kl_loss /= b_size

                if loss.data[0] != loss.data[0]: # nan detection
                    log.info("***** UH OH NAN DETECTED *****")
                    pdb.set_trace()

                # backward + optimization
                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    if args.max_grad_norm:
                        grad_norm = clip_grad_norm(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    step += 1

                # bookkeeping
                tracker['ELBO'] = torch.cat((tracker['ELBO'], loss.data))
                loss = loss.data[0]
                if args.model == 'vae':
                    tracker['NLL'] = torch.cat((tracker['NLL'], nll_loss.data))
                    tracker['KL'] = torch.cat((tracker['NLL'], kl_loss.data))
                    nll_loss = nll_loss.data[0]
                    kl_loss = kl_loss.data[0]
                else:
                    tracker['NLL'] = torch.cat((tracker['NLL'], model.tensor([0])))
                    tracker['KL'] = torch.cat((tracker['KL'], model.tensor([0])))

                if args.tensorboard_logging:
                    writer.add_scalar("%s/ELBO"%split.upper(), loss, epoch*n_batches + iteration)
                    writer.add_scalar("%s/NLL Loss"%split.upper(), nll_loss, epoch*n_batches + iteration)
                    writer.add_scalar("%s/KL Loss"%split.upper(), kl_loss, epoch*n_batches + iteration)
                    writer.add_scalar("%s/KL Weight"%split.upper(), kl_weight, epoch*n_batches + iteration)

                if iteration % args.print_every == 0 or iteration + 1 == n_batches:
                    log.info("  Batch %04d/%i\tLoss %9.4f\tNLL-Loss %9.4f\tKL-Loss %9.4f\tKL-Weight %6.3f",
                        iteration, n_batches-1, loss, nll_loss, kl_loss, kl_weight)

                if split == 'valid': # store the dev target sentences
                    if 'target_sents' not in tracker:
                        tracker['target_sents'] = list()
                    tracker['target_sents'] += idx2word(batch['target'].data, \
                            i2w=datasets['train'].get_i2w(), pad_idx=datasets['train'].pad_idx)
                    if args.model == 'vae':
                        tracker['z'] = torch.cat((tracker['z'], z.data), dim=0)

            log.info("  Mean ELBO %9.4f, NLL: %9.4f", torch.mean(tracker['ELBO']), torch.mean(tracker['NLL']))

            if args.tensorboard_logging:
                writer.add_scalar("%s-Epoch/ELBO" % split.upper(), torch.mean(tracker['ELBO']), epoch)

            # save a dump of all sentences and the encoded latent space
            if split == 'valid':
                loss = torch.mean(tracker['ELBO'])
                dump = {'target_sents':tracker['target_sents'], 'z':tracker['z'].tolist()}
                if not os.path.exists(os.path.join('dumps', ts)):
                    os.makedirs('dumps/' + ts)
                with open(os.path.join('dumps/'+ ts +'/valid_E%i.json'%epoch), 'w') as dump_file:
                    json.dump(dump, dump_file)
                if loss < global_tracker['best_score'] or global_tracker['best_score'] < 0:
                    log.info("  Best model found")
                    global_tracker['best_epoch'] = epoch
                    global_tracker['best_score'] = loss
                    checkpoint_path = os.path.join(save_model_path, "best.mdl")
                    torch.save(model.state_dict(), checkpoint_path)
                if kl_weight >= 1 - EPS:
                    if len(global_tracker['history']) >= args.patience and \
                            loss >= min(global_tracker['history'][-args.patience:]):
                        log.info("Ran out of patience!")
                        stop_training = 1
                    global_tracker['history'].append(loss)
                scheduler.step(loss, epoch)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
