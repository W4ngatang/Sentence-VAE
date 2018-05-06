''' Model architectures '''
import math
import random
import ipdb as pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
#from utils import to_var

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def kl_anneal_function(anneal_function, step, k, x0):
    if anneal_function == 'logistic':
        return float(1/(1+np.exp(-k*(step-x0))))
    elif anneal_function == 'linear':
        return min(1, step/x0)

def noise_fn(sents, prob_drop, prob_swap):
    ''' Apply swap and drop noise to a batch '''
    # NOTE(Alex): may need to increment lengths for <sos>/<eos> targs
    noisy_batch = []
    for sent in sents:
        drop_noise = random.choices(range(2), weights=[1-prob_drop, prob_drop], k=len(sent))
        new_sent = [w for w, n in zip(sent, drop_noise) if not n]
        new_length = len(new_sent)

        swap_noise = random.choices(range(2), weights=[1-prob_swap, prob_swap],
                                    k=math.floor(new_length/2))
        for idx, noise in enumerate(swap_noise):
            if noise:
                new_sent[2*idx], new_sent[2*idx+1] = new_sent[2*idx+1], new_sent[2*idx]
        noisy_batch.append(new_sent)
    return noisy_batch

class SentenceVAE(nn.Module):

    def __init__(self, args, word2idx, embedding_size, rnn_type, hidden_size, word_dropout, latent_size,
                 num_layers=1, bidirectional=False):
        super(SentenceVAE, self).__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        self.word2idx = word2idx
        self.sos_idx = word2idx['<sos>'] #sos_idx
        self.eos_idx = word2idx['<eos>'] #eos_idx
        self.pad_idx = word2idx['<pad>'] #pad_idx
        vocab_size = len(word2idx)
        #self.max_sequence_length = max_sequence_length

        self.latent_size = latent_size

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.word_dropout = nn.Dropout(p=word_dropout)

        self.denoise = args.denoise
        self.prob_swap = args.prob_swap
        self.prob_drop = args.prob_drop

        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        # elif rnn_type == 'lstm':
        #     rnn = nn.LSTM
        else:
            raise ValueError()

        self.encoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers,
                               bidirectional=self.bidirectional, batch_first=True)
        self.decoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers,
                               bidirectional=False, batch_first=True)

        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        self.hidden2mean = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.hidden2logv = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.latent2hidden = nn.Linear(latent_size, hidden_size * self.hidden_factor)
        self.outputs2vocab = nn.Linear(hidden_size * (2 if bidirectional else 1), vocab_size)

    def prepare_batch(self, sentences, add_sentinels=True, denoise=True):
        ''' Prepare sentences for model

        args:
            - sentences List[List[str]]
            - add_sentinels: True if add <sos>, <eos>
        '''
        batch_size = len(sentences)
        if self.denoise and denoise:
            src_sents = noise_fn(sentences, self.prob_drop, self.prob_swap)
        else:
            src_sents = sentences
        trg_sents = sentences
        lens_and_sents = [(len(s), s, t, idx) for idx, (s, t) in \
                            enumerate(zip(src_sents, trg_sents))]
        lens_and_sents.sort(key=lambda x: x[0], reverse=True)
        src_sents = [s for _, s, _, _ in lens_and_sents]
        trg_sents = [t for _, _, t, _ in lens_and_sents]
        src_lens = [l for l, _, _, _ in lens_and_sents]
        trg_lens = [len(s) for s in trg_sents]
        max_src_len = max(src_lens)
        max_trg_len = max(trg_lens)
        unsort = [i for _, _, _, i in lens_and_sents]
        if add_sentinels:
            src_sents = [['<sos>'] + s for s in src_sents]
            src_lens = [l + 1 for l in src_lens]
            trg_sents = [s + ['<eos>'] for s in trg_sents]
            trg_lens = [l + 1 for l in trg_lens]
            max_src_len += 1
            max_trg_len += 1
        src_sents = [[self.word2idx[w] if w in self.word2idx else self.word2idx['<unk>'] \
                     for w in s] for s in src_sents]
        trg_sents = [[self.word2idx[w] if w in self.word2idx else self.word2idx['<unk>'] \
                     for w in s] for s in trg_sents]
        for sent in src_sents:
            sent += [self.word2idx['<pad>']] * (max_src_len - len(sent))
        for sent in trg_sents:
            sent += [self.word2idx['<pad>']] * (max_trg_len - len(sent))

        src_sents = Variable(self.tensor(src_sents).long())
        trg_sents = Variable(self.tensor(trg_sents).long())
        batch = {'input': src_sents, 'src_length': src_lens,
                 'target': trg_sents, 'trg_length': trg_lens,
                 'unsort': unsort}
        return batch

    def forward(self, input_sequence, trg_sequence, src_length, trg_length):
        batch_size = input_sequence.size(0)
        sorted_src_lengths, sorted_idx_src = torch.sort(src_length, descending=True)
        input_sequence = input_sequence[sorted_idx_src]
        trg_sequence = trg_sequence[sorted_idx_src]

        # ENCODER
        input_embedding = self.embedding(input_sequence)

        packed_input = rnn_utils.pack_padded_sequence(input_embedding,
                                                      sorted_src_lengths.data.tolist(),
                                                      batch_first=True)
        _, hidden = self.encoder_rnn(packed_input) # they throw away the output; (n_layers * n_dir, batch_size, d_hid)

        hidden = hidden.transpose(0, 1).contiguous().view(batch_size, -1)
        '''
        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden = hidden.view(batch_size, self.hidden_size*self.hidden_factor)
        else:
            hidden = hidden.squeeze()
        '''

        # REPARAMETERIZATION
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        z = to_var(torch.randn([batch_size, self.latent_size]))
        z = z * std + mean

        # DECODER
        hidden = self.latent2hidden(z)

        hidden = hidden.view(batch_size, self.hidden_factor, self.hidden_size)#.transpose(0, 1).contiguous()

        # decoder input
        sorted_trg_lengths, sorted_idx_trg = torch.sort(trg_length, descending=True)
        trg_sequence = trg_sequence[sorted_idx_trg]
        hidden = hidden[sorted_idx_trg]
        hidden = hidden.transpose(0, 1).contiguous()
        '''
        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)
        '''

        input_embedding = self.word_dropout(input_embedding)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding,
                                                      sorted_trg_lengths.data.tolist(),
                                                      batch_first=True)

        # decoder forward pass
        outputs, _ = self.decoder_rnn(packed_input, hidden)

        # process outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        _, reversed_idx_trg = torch.sort(sorted_idx_trg)
        _, reversed_idx_src = torch.sort(sorted_idx_src)
        padded_outputs = padded_outputs[reversed_idx_trg]
        padded_outputs = padded_outputs[reversed_idx_src]
        b, s,_ = padded_outputs.size()

        # project outputs to vocab
        logp = nn.functional.log_softmax(self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
        logp = logp.view(b, s, self.embedding.num_embeddings)


        return logp, mean, logv, z

    def loss_fn(self, logp, target, length, mean, logv, anneal_function, step, k, x0):
        # cut-off unnecessary padding from target, and flatten
        target = target[:, :torch.max(length).data[0]].contiguous().view(-1)
        logp = logp.view(-1, logp.size(2))

        # Negative Log Likelihood
        #NLL = torch.nn.NLLLoss(size_average=False, ignore_index=datasets['train'].pad_idx)
        nll_loss = F.nll_loss(logp, target, size_average=False, ignore_index=self.pad_idx)

        # KL Divergence
        kl_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        kl_weight = kl_anneal_function(anneal_function, step, k, x0)

        return nll_loss, kl_loss, kl_weight


    def encode(self, sentences, batch_size=64):
        ''' Given a batch of input sequences, return mean and std_dev vectors '''
        means, std_devs = [], []
        for b_idx in range(0, len(sentences), batch_size):
            batch = self.prepare_batch(sentences[b_idx*batch_size:(b_idx+1)*batch_size],
                                       denoise=False)

            # ENCODE
            input_embedding = self.embedding(batch['input'])
            packed_input = rnn_utils.pack_padded_sequence(input_embedding, batch['src_length'],
                                                          batch_first=True)
            _, hidden = self.encoder_rnn(packed_input)

            if self.bidirectional or self.num_layers > 1:
                # flatten hidden state
                hidden = hidden.view(batch_size, self.hidden_size*self.hidden_factor)
            else:
                hidden = hidden.squeeze(0) # I think

            # REPARAMETERIZATION
            mean = self.hidden2mean(hidden).data.cpu().numpy()
            logv = self.hidden2logv(hidden)
            std = torch.exp(0.5 * logv).data.cpu().numpy()
            idx_unsort = np.argsort(batch['unsort'])
            means.append(mean[idx_unsort])
            std_devs.append(std[idx_unsort])
        means = np.vstack(means)
        std_devs = np.vstack(means)
        return means#, std_devs


    def inference(self, n=4, z=None, max_seq_len=50):
        # TODO(Alex): wtf is this

        if z is None:
            batch_size = n
            z = to_var(torch.randn([batch_size, self.latent_size]))
        else:
            batch_size = z.size(0)

        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)

        hidden = hidden.unsqueeze(0)

        # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch
        sequence_running = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch which are still generating
        sequence_mask = torch.ones(batch_size, out=self.tensor()).byte()

        running_seqs = torch.arange(0, batch_size, out=self.tensor()).long() # idx of still generating sequences with respect to current loop

        generations = self.tensor(batch_size, max_seq_len).fill_(self.pad_idx).long()

        t = 0
        while(t < max_seq_len and len(running_seqs)>0):

            if t == 0:
                input_sequence = to_var(torch.Tensor(batch_size).fill_(self.sos_idx).long())

            input_sequence = input_sequence.unsqueeze(1)

            input_embedding = self.embedding(input_sequence)

            output, hidden = self.decoder_rnn(input_embedding, hidden)

            logits = self.outputs2vocab(output)

            input_sequence = self._sample(logits)

            # save next input
            generations = self._save_sample(generations, input_sequence, sequence_running, t)

            # update gloabl running sequence
            sequence_mask[sequence_running] = (input_sequence != self.eos_idx).data
            sequence_running = sequence_idx.masked_select(sequence_mask)

            # update local running sequences
            running_mask = (input_sequence != self.eos_idx).data
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                input_sequence = input_sequence[running_seqs]
                hidden = hidden[:, running_seqs]

                running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()

            t += 1

        return generations, z

    def _sample(self, dist, mode='greedy'):

        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.squeeze()

        return sample

    def _save_sample(self, save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:,t] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to

class SentenceAE(nn.Module):

    def __init__(self, args, word2idx, embedding_size, rnn_type, hidden_size, word_dropout, latent_size,
                 num_layers=1, bidirectional=False):
        super(SentenceAE, self).__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        self.word2idx = word2idx
        self.sos_idx = word2idx['<sos>'] #sos_idx
        self.eos_idx = word2idx['<eos>'] #eos_idx
        self.pad_idx = word2idx['<pad>'] #pad_idx
        vocab_size = len(word2idx)
        #self.max_sequence_length = max_sequence_length

        self.latent_size = latent_size

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.word_dropout = nn.Dropout(p=word_dropout)

        self.denoise = args.denoise
        self.prob_swap = args.prob_swap
        self.prob_drop = args.prob_drop

        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        # elif rnn_type == 'lstm':
        #     rnn = nn.LSTM
        else:
            raise ValueError()

        self.encoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers,
                               bidirectional=self.bidirectional, batch_first=True)
        self.decoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers,
                               bidirectional=False, batch_first=True)
        self.hidden_factor = (2 if bidirectional else 1) * num_layers
        self.outputs2vocab = nn.Linear(hidden_size * (2 if bidirectional else 1), vocab_size)

    def prepare_batch(self, sentences, add_sentinels=True, denoise=True):
        ''' Prepare sentences for model

        args:
            - sentences List[List[str]]
            - add_sentinels: True if add <sos>, <eos>
        '''
        batch_size = len(sentences)
        if self.denoise and denoise:
            src_sents = noise_fn(sentences, self.prob_drop, self.prob_swap)
        else:
            src_sents = sentences
        trg_sents = sentences
        lens_and_sents = [(len(s), s, t, idx) for idx, (s, t) in \
                            enumerate(zip(src_sents, trg_sents))]
        lens_and_sents.sort(key=lambda x: x[0], reverse=True)
        src_sents = [s for _, s, _, _ in lens_and_sents]
        trg_sents = [t for _, _, t, _ in lens_and_sents]
        src_lens = [l for l, _, _, _ in lens_and_sents]
        trg_lens = [len(s) for s in trg_sents]
        max_src_len = max(src_lens)
        max_trg_len = max(trg_lens)
        unsort = [i for _, _, _, i in lens_and_sents]
        if add_sentinels:
            src_sents = [['<sos>'] + s for s in src_sents]
            src_lens = [l + 1 for l in src_lens]
            trg_sents = [s + ['<eos>'] for s in trg_sents]
            trg_lens = [l + 1 for l in trg_lens]
            max_src_len += 1
            max_trg_len += 1
        src_sents = [[self.word2idx[w] if w in self.word2idx else self.word2idx['<unk>'] \
                     for w in s] for s in src_sents]
        trg_sents = [[self.word2idx[w] if w in self.word2idx else self.word2idx['<unk>'] \
                     for w in s] for s in trg_sents]
        for sent in src_sents:
            sent += [self.word2idx['<pad>']] * (max_src_len - len(sent))
        for sent in trg_sents:
            sent += [self.word2idx['<pad>']] * (max_trg_len - len(sent))

        src_sents = Variable(self.tensor(src_sents).long())
        trg_sents = Variable(self.tensor(trg_sents).long())

        batch = {'input': src_sents, 'src_length': src_lens,
                 'target': trg_sents, 'trg_length': trg_lens,
                 'unsort': unsort}
        return batch

    def forward(self, input_sequence, trg_sequence, src_length, trg_length):
        batch_size = input_sequence.size(0)
        sorted_src_lengths, sorted_idx_src = torch.sort(src_length, descending=True)
        input_sequence = input_sequence[sorted_idx_src]
        trg_sequence = trg_sequence[sorted_idx_src]

        # ENCODER
        input_embedding = self.embedding(input_sequence)

        packed_input = rnn_utils.pack_padded_sequence(input_embedding,
                                                      sorted_src_lengths.data.tolist(),
                                                      batch_first=True)
        _, hidden = self.encoder_rnn(packed_input) # they throw away the output; (n_layers * n_dir, batch_size, d_hid)
        '''
        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden = hidden.view(batch_size, self.hidden_size*self.hidden_factor)
        else:
            hidden = hidden.squeeze()
        '''
        hidden = hidden.view(batch_size, self.hidden_factor, self.hidden_size)#.transpose(0, 1).contiguous()

        # decoder input
        sorted_trg_lengths, sorted_idx_trg = torch.sort(trg_length, descending=True)
        trg_sequence = trg_sequence[sorted_idx_trg]
        hidden = hidden[sorted_idx_trg]
        hidden = hidden.transpose(0, 1).contiguous()
        '''
        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)
        '''

        trg_embedding = self.embedding(trg_sequence)
        trg_embedding = self.word_dropout(trg_embedding)
        packed_input = rnn_utils.pack_padded_sequence(trg_embedding,
                                                      sorted_trg_lengths.data.tolist(),
                                                      batch_first=True)

        # decoder forward pass
        outputs, _ = self.decoder_rnn(packed_input, hidden)

        # process outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()

        _, reversed_idx_trg = torch.sort(sorted_idx_trg)
        _, reversed_idx_src = torch.sort(sorted_idx_src)
        padded_outputs = padded_outputs[reversed_idx_trg]
        padded_outputs = padded_outputs[reversed_idx_src]
        b, s,_ = padded_outputs.size()

        # project outputs to vocab
        logp = nn.functional.log_softmax(self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
        logp = logp.view(b, s, self.embedding.num_embeddings)

        return logp, hidden, None, None

    def loss_fn(self, logp, target, length, mean, logv, anneal_function, step, k, x0):
        # cut-off unnecessary padding from target, and flatten
        target = target[:, :torch.max(length).data[0]].contiguous().view(-1)
        logp = logp.view(-1, logp.size(2))

        # Negative Log Likelihood
        #NLL = torch.nn.NLLLoss(size_average=False, ignore_index=datasets['train'].pad_idx)
        nll_loss = F.nll_loss(logp, target, size_average=False, ignore_index=self.pad_idx)

        # KL Divergence
        kl_loss = 0. #-0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        kl_weight = 0. #kl_anneal_function(anneal_function, step, k, x0)

        return nll_loss, kl_loss, kl_weight

    def encode(self, sentences, batch_size=64):
        ''' Given a batch of input sequences, return mean and std_dev vectors '''
        hiddens = []
        for b_idx in range(0, len(sentences), batch_size):
            batch = self.prepare_batch(sentences[b_idx*batch_size:(b_idx+1)*batch_size],
                                       denoise=False)

            # ENCODE
            input_embedding = self.embedding(batch['input'])
            packed_input = rnn_utils.pack_padded_sequence(input_embedding, batch['src_length'],
                                                          batch_first=True)
            _, hidden = self.encoder_rnn(packed_input)

            if self.bidirectional or self.num_layers > 1:
                # flatten hidden state
                hidden = hidden.view(batch_size, self.hidden_size*self.hidden_factor)
            else:
                hidden = hidden.squeeze(0) # I think

            idx_unsort = np.argsort(batch['unsort'])
            hidden = hidden.data.cpu().numpy()
            hiddens.append(hidden[idx_unsort])
        hiddens = np.vstack(hiddens)
        return hidden

    def inference(self, n=4, z=None, max_seq_len=50):

        if z is None:
            batch_size = n
            z = to_var(torch.randn([batch_size, self.latent_size]))
        else:
            batch_size = z.size(0)

        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)

        hidden = hidden.unsqueeze(0)

        # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch
        sequence_running = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch which are still generating
        sequence_mask = torch.ones(batch_size, out=self.tensor()).byte()

        running_seqs = torch.arange(0, batch_size, out=self.tensor()).long() # idx of still generating sequences with respect to current loop

        generations = self.tensor(batch_size, max_seq_len).fill_(self.pad_idx).long()

        t = 0
        while(t < max_seq_len and len(running_seqs)>0):

            if t == 0:
                input_sequence = to_var(torch.Tensor(batch_size).fill_(self.sos_idx).long())

            input_sequence = input_sequence.unsqueeze(1)

            input_embedding = self.embedding(input_sequence)

            output, hidden = self.decoder_rnn(input_embedding, hidden)

            logits = self.outputs2vocab(output)

            input_sequence = self._sample(logits)

            # save next input
            generations = self._save_sample(generations, input_sequence, sequence_running, t)

            # update gloabl running sequence
            sequence_mask[sequence_running] = (input_sequence != self.eos_idx).data
            sequence_running = sequence_idx.masked_select(sequence_mask)

            # update local running sequences
            running_mask = (input_sequence != self.eos_idx).data
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                input_sequence = input_sequence[running_seqs]
                hidden = hidden[:, running_seqs]

                running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()

            t += 1

        return generations, z

    def _sample(self, dist, mode='greedy'):

        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.squeeze()

        return sample

    def _save_sample(self, save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:,t] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to
