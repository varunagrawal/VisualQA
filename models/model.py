import torch
from torch import nn
from torch.autograd import Variable



class DeeperLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=300,
                 image_dim=4096, image_embed_dim=1024,
                 hidden_dim=2048, rnn_output_dim=1024,
                 output_dim=1000):
        """

        :param vocab_size:
        :param embed_dim:
        :param image_dim:
        :param image_embed_dim:
        :param hidden_dim:
        :param rnn_output_dim:
        :param output_dim:
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
            nn.Linear(rnn_output_dim, output_dim),
            nn.Tanh())

    def _init_hidden(self, q):
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
