"""
A baseline CNN + LSTM model as detailed in the VQA paper by Agrawal et. al.
"""

import torch
from torch import nn
from torch.autograd import Variable



class DeeperLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=300,
                 image_dim=4096, image_embed_dim=1024,
                 hidden_dim=512, rnn_output_dim=1024,
                 output_dim=1000):
        """

        :param vocab_size: The number of words in the vocabulary
        :param embed_dim: The question embedding dimensionality
        :param image_dim: The image feature dimensionality
        :param image_embed_dim: The image embedding dimensionality
        :param hidden_dim: The dimensionality of the RNN's hidden state
        :param rnn_output_dim: The RNN output dimensionality
        :param output_dim: The number of answers to output over.
        """
        super().__init__()

        self.hidden_dim = hidden_dim

        self.image_embed = nn.Sequential(
            nn.Linear(image_dim, image_embed_dim),
            nn.Tanh())

        # The question is of the format Batch x T x one-hot vector of size vocab_size
        self.embedding = nn.Sequential(
            nn.Linear(vocab_size, embed_dim),
            nn.Dropout(p=0.5),
            nn.Tanh())

        self.num_rnn_layers = 2
        self.num_directions = 1

        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=self.num_rnn_layers)
        self.fc = nn.Linear(self.num_rnn_layers * hidden_dim, rnn_output_dim)
        self.activ = nn.Tanh()

        self.mlp = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(rnn_output_dim, output_dim))

    def _init_hidden(self, q):
        """
        Initialize the hidden state of the RNN
        :param q: The question embedding. Used for getting the hidden state dimensions.
        :return: The initial hidden state for the RNN
        """
        hidden = [Variable(torch.zeros(self.num_rnn_layers*self.num_directions, q.size(1), self.hidden_dim)),
                  Variable(torch.zeros(self.num_rnn_layers*self.num_directions, q.size(1), self.hidden_dim))]
        if torch.cuda.is_available():
            hidden = [x.cuda() for x in hidden]

        return hidden

    def forward(self, img, ques):
        img_features = self.image_embed(img)

        q = self.embedding(ques)  # BxTxD

        q = q.transpose(0, 1)  # makes this TxBxD

        # initialize the hidden state for each mini-batch
        hidden = self._init_hidden(q)
        lstm_out, hidden = self.rnn(q, hidden)

        hidden_state, cell = hidden

        # convert from TxBxD to BxTxD and make contiguous
        hidden_state =  hidden_state.transpose(0, 1).contiguous()
        # Make from [B, n_layers, hidden_dim] to [B, n_layers*hidden_dim]
        hidden_state = hidden_state.view(hidden_state.size(0), -1)

        x = self.fc(hidden_state)
        x = self.activ(x)

        # Fusion
        x = x.mul(img_features)

        output = self.mlp(x)

        return output
