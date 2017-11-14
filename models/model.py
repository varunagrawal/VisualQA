import enum
from . import deeperlstm, deeper_embed_lstm

class Models(enum.Enum):
    DeeperLSTM = deeperlstm.DeeperLSTM
    DeeperEmbedLSTM = deeper_embed_lstm.DeeperEmbedLSTM

