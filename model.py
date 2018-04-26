''' Model architectures

TODO:
    - SentencEAE

'''
import ipdb as pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
#from utils import to_var

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

class SentenceVAE(nn.Module):

    #def __init__(self, vocab_size, embedding_size, rnn_type, hidden_size, word_dropout, latent_size,
    #            sos_idx, eos_idx, pad_idx, max_sequence_length, num_layers=1, bidirectional=False):
    def __init__(self, word2idx, embedding_size, rnn_type, hidden_size, word_dropout, latent_size,
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

        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        # elif rnn_type == 'lstm':
        #     rnn = nn.LSTM
        else:
            raise ValueError()

        self.encoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)
        self.decoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)

        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        self.hidden2mean = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.hidden2logv = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.latent2hidden = nn.Linear(latent_size, hidden_size * self.hidden_factor)
        self.outputs2vocab = nn.Linear(hidden_size * (2 if bidirectional else 1), vocab_size)

    def prepare_batch(self, sentences, add_sentinels=True):
        ''' Prepare sentences for model

        args:
            - sentences List[List[str]]
            - add_sentinels: True if add <sos>, <eos>
        '''
        batch_size = len(sentences)
        #max_len = max([len(s) for s in sentences])
        lens_and_sents = [(len(s), s, idx) for idx, s in enumerate(sentences)]
        lens_and_sents.sort(key=lambda x: x[0], reverse=True)
        sentences = [s for _, s, _ in lens_and_sents]
        lengths = [l for l, _, _ in lens_and_sents]
        max_len = lengths[0]
        unsort = [i for _, _, i in lens_and_sents]
        if add_sentinels:
            sentences = [['<sos>'] + s + ['<eos>'] for s in sentences]
            lengths = [l + 2 for l in lengths]
            max_len += 2
        idx_sents = [[self.word2idx[w] if w in self.word2idx else self.word2idx['<unk>'] for w in s] for s in sentences]
        for idx_sent in idx_sents:
            idx_sent += [self.word2idx['<pad>']] * (max_len - len(idx_sent))
        batch = Variable(self.tensor(idx_sents).long())
        return batch, lengths, unsort

    def forward(self, input_sequence, length):
        batch_size = input_sequence.size(0)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input_sequence[sorted_idx]

        # ENCODER
        input_embedding = self.embedding(input_sequence)

        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)
        _, hidden = self.encoder_rnn(packed_input)

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden = hidden.view(batch_size, self.hidden_size*self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        # REPARAMETERIZATION
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        z = to_var(torch.randn([batch_size, self.latent_size]))
        z = z * std + mean

        # DECODER
        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        # decoder input
        input_embedding = self.word_dropout(input_embedding)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        # decoder forward pass
        outputs, _ = self.decoder_rnn(packed_input, hidden)

        # process outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        _,reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        b,s,_ = padded_outputs.size()

        # project outputs to vocab
        logp = nn.functional.log_softmax(self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
        logp = logp.view(b, s, self.embedding.num_embeddings)


        return logp, mean, logv, z

    def encode(self, sentences, batch_size=64):
        ''' Given a batch of input sequences, return mean and std_dev vectors '''
        means, std_devs = [], []
        for b_idx in range(0, len(sentences), batch_size):
            batch, lengths, unsort = self.prepare_batch(sentences[b_idx*batch_size:(b_idx+1)*batch_size])

            # ENCODE
            input_embedding = self.embedding(batch)
            packed_input = rnn_utils.pack_padded_sequence(input_embedding, lengths, batch_first=True)
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
            idx_unsort = np.argsort(unsort)
            means.append(mean[idx_unsort])
            std_devs.append(std[idx_unsort])
        means = np.vstack(means)
        std_devs = np.vstack(means)
        return means, std_devs


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

        generations = self.tensor(batch_size, self.max_seq_len).fill_(self.pad_idx).long()

        t=0
        while(t<self.max_seq_len and len(running_seqs)>0):

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
