import os
import json
import torch
import argparse

from model import SentenceVAE, SentenceAE
from utils import to_var, idx2word, interpolate


def main(args):

    with open(args.data_dir+'/vocab.json', 'r') as file:
        vocab = json.load(file)

    w2i, i2w = vocab['w2i'], vocab['i2w']

    if args.model == 'vae':
        model = SentenceVAE(args, w2i,
            embedding_size=args.embedding_size,
            rnn_type=args.rnn_type,
            hidden_size=args.hidden_size,
            word_dropout=args.word_dropout,
            latent_size=args.latent_size,
            num_layers=args.num_layers,
            bidirectional=args.bidirectional)
    elif args.model == 'ae':
        model = SentenceAE(args, w2i,
            embedding_size=args.embedding_size,
            rnn_type=args.rnn_type,
            hidden_size=args.hidden_size,
            word_dropout=args.word_dropout,
            latent_size=args.latent_size,
            num_layers=args.num_layers,
            bidirectional=args.bidirectional)

    if not os.path.exists(args.load_checkpoint):
        raise FileNotFoundError(args.load_checkpoint)

    model.load_state_dict(torch.load(args.load_checkpoint))
    print("Model loaded from %s"%(args.load_checkpoint))

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    samples, z = model.inference(n=args.num_samples, max_seq_len=args.max_sequence_length)
    print('----------SAMPLES----------')
    print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')

    z1 = torch.randn([args.latent_size]).numpy()
    z2 = torch.randn([args.latent_size]).numpy()
    z = to_var(torch.from_numpy(interpolate(start=z1, end=z2, steps=8)).float())
    samples, _ = model.inference(z=z)
    print('-------INTERPOLATION-------')
    print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--load_checkpoint', type=str)
    parser.add_argument('-n', '--num_samples', type=int, default=10)

    parser.add_argument('-dd', '--data_dir', type=str, default='data')
    parser.add_argument('-ms', '--max_sequence_length', type=int, default=50)

    parser.add_argument('-m', '--model', type=str, default='vae')
    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=512)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0.5)
    parser.add_argument('-ls', '--latent_size', type=int, default=16)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')

    parser.add_argument('-d', '--denoise', action='store_true')
    parser.add_argument('-pd', '--prob_drop', type=float, default=0.1)
    parser.add_argument('-ps', '--prob_swap', type=float, default=0.1)

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']

    main(args)
