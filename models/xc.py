"""
A VQA model using eXtreme Classification.
"""

import torch
from torch import nn
from torch.nn import utils
from VisualQA.models import extractor


class VQAX(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, image_dim=4096,
                 image_embed_dim=1024, ques_hidden_dim=512, rnn_output_dim=1024,
                 hidden_dim=512, output_dim=1000, batch_first=True, raw_images=False):
        """

        :param vocab_size: The number of words in the vocabulary
        :param embed_dim: The question embedding dimensionality
        :param image_dim: The image feature dimensionality
        :param image_embed_dim: The image embedding dimensionality
        :param ques_hidden_dim: The dimensionality of the RNN's hidden state
        :param rnn_output_dim: The RNN output dimensionality
        :param hidden_dim: The hidden layer to compress the model
        :param output_dim: The number of answers to output over.
        :param batch_first: Flag to indicate if the RNN accepts input with batch dim leading.
        """
        super().__init__()

        self.raw_images = raw_images
        if raw_images:
            # base model uses VGG19
            self.feature_extractor = extractor.FeatureExtractor("vgg19_bn")

        self.image_embed = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(image_dim, image_embed_dim),
            nn.ReLU())

        # add 1 to account for padding value
        vocab_size = vocab_size + 1

        self.embedding = nn.Sequential(
            nn.Embedding(vocab_size, embed_dim, padding_idx=0),
            nn.Dropout(p=0.5),
            nn.ReLU())

        self.num_rnn_layers = 2
        self.num_directions = 1
        self.batch_first = batch_first

        self.rnn = nn.LSTM(embed_dim, ques_hidden_dim,
                           num_layers=self.num_rnn_layers,
                           batch_first=self.batch_first)

        self.question_embed = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.num_rnn_layers * 2 * ques_hidden_dim,
                      rnn_output_dim),  # 2 for hidden + cell
            nn.ReLU()
        )

        mlp_input_dim = image_embed_dim + rnn_output_dim

        self.mlp = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, output_dim)
        )

        self.init_weights()

    def init_weights(self):
        for l in self.embedding:
            if hasattr(l, 'weight'):
                nn.init.uniform_(l.weight, -0.08, 0.08)

        # nn.init.uniform_(self.rnn.weight_ih_l0, -0.08, 0.08)
        # nn.init.uniform_(self.rnn.weight_hh_l0, -0.08, 0.08)

        for l in self.image_embed:
            if hasattr(l, 'weight'):
                nn.init.uniform_(l.weight, -0.08, 0.08)

        for l in self.question_embed:
            if hasattr(l, 'weight'):
                nn.init.uniform_(l.weight, -0.08, 0.08)

        for l in self.mlp:
            if hasattr(l, 'weight'):
                nn.init.uniform_(l.weight, -0.08, 0.08)

    def forward(self, img, ques, q_lens):
        if self.raw_images:
            img_feat = self.feature_extractor(img)
        else:
            img_feat = img

        # normalize the image and embed it to the common dimension.
        norm = img_feat.norm(p=2, dim=1, keepdim=True).expand_as(img_feat)
        img_feat_norm = img_feat / norm.detach()

        q = self.embedding(ques)  # BxTxD

        # initial hidden state defaults to 0
        output, (hidden_state, cell) = self.rnn(q)

        # convert from NxBxD to BxNxD and make contiguous, where N is the number of layers in the RNN
        hidden_state, cell = hidden_state.transpose(0, 1).contiguous(), \
            cell.transpose(0, 1).contiguous()
        # Make from [B, n_layers, hidden_dim] to [B, n_layers*hidden_dim]
        hidden_state, cell = hidden_state.view(hidden_state.size(0), -1), \
            cell.view(cell.size(0), -1)
        # Concatenate the hidden state and the cell state to get the question embedding
        q_embed = torch.cat((hidden_state, cell), dim=1)

        # map both modalities to common space
        img_features = self.image_embed(img_feat_norm)
        ques_features = self.question_embed(q_embed)

        # Fusion
        x = torch.cat((img_features, ques_features), dim=1)

        # Classification
        output = self.mlp(x)

        return output
