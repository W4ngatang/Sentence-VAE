''' Sentence VAE '''
import os
import sys
import json
import time
import random
import argparse
import ipdb as pdb
import logging as log

import torch
import torch.optim as optim
import numpy as np
from multiprocessing import cpu_count
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict

from ptb import PTB
from utils import to_var, idx2word, expierment_name
from model import SentenceVAE

SCR_PREFIX = '/misc/vlgscratch4/BowmanGroup/awang/'

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
    NLL_loss = NLL(logp, target)

    # KL Divergence
    KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
    KL_weight = kl_anneal_function(anneal_function, step, k, x0)

    return NLL_loss, KL_loss, KL_weight

def main(arguments):
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', help='random seed', type=int, default=19)
    parser.add_argument('--run_name', help='prefix to save ckpts to', type=str,
                        default=SCR_PREFIX + 'ckpts/svae/test/')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('--max_sequence_length', type=int, default=60)
    parser.add_argument('--min_occ', type=int, default=1)
    parser.add_argument('--test', action='store_true')

    parser.add_argument('-ep', '--epochs', type=int, default=20)
    parser.add_argument('-bs', '--batch_size', type=int, default=128)
    parser.add_argument('-o', '--optimizer', type=str, choices=['sgd', 'adam'], default='adam')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('--lr_decay_factor', type=float, default=0.5)
    parser.add_argument('-p', '--patience', type=int, default=5)

    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, choices=['rnn', 'lstm', 'gru'],
                        default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')
    parser.add_argument('-ls', '--latent_size', type=int, default=16)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0.5)

    parser.add_argument('-af', '--anneal_function', type=str, choices=['logistic', 'linear'],
                        default='logistic')
    parser.add_argument('-k', '--k', type=float, default=0.0025)
    parser.add_argument('-x0', '--x0', type=int, default=2500)

    parser.add_argument('-v', '--print_every', type=int, default=50)
    parser.add_argument('-tb', '--tensorboard_logging', action='store_true')
    parser.add_argument('-log', '--logdir', type=str, default='logs')
    parser.add_argument('-bin', '--save_model_path', type=str, default='bin')

    args = parser.parse_args(arguments)

    log.basicConfig(format="%(asctime)s: %(message)s", level=log.INFO, datefmt='%m/%d %I:%M:%S %p')
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

    model = SentenceVAE(datasets['train'].get_w2i(),
        embedding_size=args.embedding_size,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional)
    if torch.cuda.is_available():
        model = model.cuda()
    log.info(model)

    if args.tensorboard_logging:
        writer = SummaryWriter(os.path.join(args.logdir, expierment_name(args,ts)))
        writer.add_text("model", str(model))
        writer.add_text("args", str(args))
        writer.add_text("ts", ts)
    save_model_path = os.path.join(args.save_model_path, args.run_name)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    NLL = torch.nn.NLLLoss(size_average=False, ignore_index=datasets['train'].pad_idx)
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=args.lr_decay_factor,
                                                     patience=0,
                                                     verbose=True)
    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    step, stop_training = 0, 0
    global_tracker = {'best_epoch': -1, 'best_score': -1, 'history': []}
    for epoch in range(args.epochs):
        if stop_training:
            break
        for split in splits:
            data_loader = DataLoader(
                dataset=datasets[split],
                batch_size=args.batch_size,
                shuffle=split=='train',
                num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available()
            )

            tracker = defaultdict(tensor)

            # Enable/Disable Dropout
            if split == 'train':
                log.info("***** Epoch %02d *****", epoch)
                log.info("Training...")
                model.train()
            else:
                log.info("Validating...")
                model.eval()

            for iteration, batch in enumerate(data_loader):
                batch_size = batch['input'].size(0)
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = to_var(v)

                # Forward pass
                logp, mean, logv, z = model(batch['input'], batch['length'])

                # loss calculation
                NLL_loss, KL_loss, KL_weight = loss_fn(NLL, logp, batch['target'],
                    batch['length'], mean, logv, args.anneal_function, step, args.k, args.x0)
                loss = (NLL_loss + KL_weight * KL_loss) / batch_size
                NLL_loss /= batch_size
                KL_loss /= batch_size

                # backward + optimization
                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1

                # bookkeepeing
                tracker['ELBO'] = torch.cat((tracker['ELBO'], loss.data))
                tracker['NLL'] = torch.cat((tracker['NLL'], NLL_loss.data))
                tracker['KL'] = torch.cat((tracker['NLL'], KL_loss.data))
                loss = loss.data[0]
                NLL_loss = NLL_loss.data[0]
                KL_loss = KL_loss.data[0]

                if args.tensorboard_logging:
                    writer.add_scalar("%s/ELBO"%split.upper(), loss, epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/NLL Loss"%split.upper(), NLL_loss, epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/KL Loss"%split.upper(), KL_loss, epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/KL Weight"%split.upper(), KL_weight, epoch*len(data_loader) + iteration)

                if iteration % args.print_every == 0 or iteration + 1 == len(data_loader):
                    log.info("  Batch %04d/%i\tLoss %9.4f\tNLL-Loss %9.4f\tKL-Loss %9.4f\tKL-Weight %6.3f",
                        iteration, len(data_loader)-1, loss, NLL_loss, KL_loss, KL_weight)

                if split == 'valid': # store the dev sentences?
                    if 'target_sents' not in tracker:
                        tracker['target_sents'] = list()
                    tracker['target_sents'] += idx2word(batch['target'].data, \
                            i2w=datasets['train'].get_i2w(), pad_idx=datasets['train'].pad_idx)
                    tracker['z'] = torch.cat((tracker['z'], z.data), dim=0)

            log.info("  Mean ELBO %9.4f, NLL: %9.4f", torch.mean(tracker['ELBO']), torch.mean(tracker['NLL']))

            if args.tensorboard_logging:
                writer.add_scalar("%s-Epoch/ELBO" % split.upper(), torch.mean(tracker['ELBO']), epoch)

            # save a dump of all sentences and the encoded latent space
            if split == 'valid':
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
                if KL_weight >= 1:
                    if len(global_tracker['history']) >= args.patience and \
                            loss > min(global_tracker['history'][-args.patience:]):
                        log.info("Ran out of patience!")
                        stop_training = 1
                    global_tracker['history'].append(loss)
                    scheduler.step(loss, epoch)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
