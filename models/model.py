import enum
from . import deeperlstm, deeper_embed_lstm, mcb, xc


class Models(enum.Enum):
    DeeperLSTM = deeperlstm.DeeperLSTM
    DeeperEmbedLSTM = deeper_embed_lstm.DeeperEmbedLSTM
    MCBModel = mcb.MCBModel
    VQAX = xc.VQAX

