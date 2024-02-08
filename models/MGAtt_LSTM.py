import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.layer import MGAtt


class MGAtt_LSTM(nn.Module):
    def __init__(self, args, graph):
        super(MGAtt_LSTM, self).__init__()
        self.args = args
        # MGAtt configure
        self.MGAtt_dropout = nn.Dropout(args.dropout)
        # MGATT model
        self.MGAtt = MGAtt(graph, args.matrix_weight, args.attention, args.M, args.d, args.bn_decay, args.feature_dim)
        # LSTM model
        self.encoder = nn.LSTM(args.city_num * args.city_num, args.lstm_hidden_size, args.lstm_layers, batch_first=True, bidirectional=False)
        self.decoder = nn.LSTM(args.lstm_hidden_size, args.lstm_hidden_size * 2, args.lstm_layers, batch_first=True)
        # fully connection layer 
        self.fully_connect = nn.Linear(args.lstm_hidden_size * 2, args.pred_len * args.city_num)
    
    def forward(self, input_x):
        # # -----------------------------------
        """ MGAtt Part """
        x = self.MGAtt(input_x)
        x = self.MGAtt_dropout(x)
        # # ------------------------------------

        # # ------------------------------------
        """ LSTM encoder-decoder Part """
        input_lstm = x.view(x.shape[0],x.shape[1], -1)
        encoder_seq, _ = self.encoder(input_lstm)
        decoder_seq, (out_h, out_c) = self.decoder(encoder_seq)
        # # ------------------------------------

        # # ------------------------------------
        """ Fully Connect Part """
        input_tail = decoder_seq[:, -1, :]
        output = self.fully_connect(input_tail)
        fc_out = output.view(-1, self.args.pred_len, self.args.city_num)
        # # ------------------------------------
        return fc_out