"""
A baseline CNN + LSTM model as detailed in the VQA paper by Agrawal et. al.
"""

import torch
from torch import nn
from torch.nn.utils import rnn
from models import extractor


class DeeperLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=300,
                 image_dim=4096, image_embed_dim=1024,
                 hidden_dim=512, rnn_output_dim=1024,
                 output_dim=1000, raw_images=False):
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

        self.raw_images = raw_images
        if raw_images:
            self.feature_extractor = extractor.FeatureExtractor("vgg16")  # base model uses VGG16

        self.hidden_dim = hidden_dim

        self.image_embed = nn.Sequential(
            nn.Linear(image_dim, image_embed_dim),
            # nn.Dropout(p=0.5),
            nn.Tanh())

        # The question is of the format Batch x T x one-hot vector of size vocab_size
        self.embedding = nn.Sequential(
            nn.Linear(vocab_size, embed_dim),
            nn.Dropout(p=0.5),
            nn.Tanh())

        self.num_rnn_layers = 2
        self.num_directions = 1

        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=self.num_rnn_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(self.num_rnn_layers * 2 * hidden_dim, rnn_output_dim)  # 2 for hidden + cell
        self.tanh = nn.Tanh()

        self.mlp = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(rnn_output_dim, output_dim),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, img, ques, q_lens):
        if self.raw_images:
            img_feat = self.feature_extractor(img)
        else:
            img_feat = img

        # normalize the image and embed it to the common dimension.
        img_feat_norm = img_feat / torch.norm(img_feat, p=2).detach()
        img_features = self.image_embed(img_feat_norm)

        q = self.embedding(ques)  # BxTxD

        # Get PackedSequence
        _, sorted_inds = torch.sort(q_lens, descending=True)
        q = q[sorted_inds]
        q_lens = q_lens[sorted_inds]
        q = rnn.pack_padded_sequence(q, q_lens, batch_first=True)
        
        # ignore outputs as we only need the embedding with dim NxBxD
        _, (hidden_state, cell) = self.rnn(q)  # initial hidden state defaults to 0
        
        # convert from NxBxD to BxNxD and make contiguous, where N is the number of layers in the RNN
        hidden_state, cell =  hidden_state.transpose(0, 1).contiguous(), cell.transpose(0, 1).contiguous()
        # Make from [B, n_layers, hidden_dim] to [B, n_layers*hidden_dim]
        hidden_state, cell = hidden_state.view(hidden_state.size(0), -1), cell.view(cell.size(0), -1)
        # Concatenate the hidden state and the cell state to get the question embedding
        q_embed = torch.cat((hidden_state, cell), dim=1)

        x = self.fc(q_embed)
        x = self.tanh(x)

        # Fusion
        x = x.mul(img_features)

        output = self.mlp(x)

        return output
