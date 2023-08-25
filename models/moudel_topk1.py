# -*- coding: utf-8 -*-#

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.parameter import Parameter
import numpy as np
from utils.process import normalize_adj


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.matmul(input, self.W)
        B, N = h.size()[0], h.size()[1]

        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, -1), h.repeat(1, N, 1)], dim=2).view(B, N, -1,
                                                                                                   2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, nlayers=2):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.nlayers = nlayers
        self.nheads = nheads
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        if self.nlayers > 2:
            for i in range(self.nlayers - 2):
                for j in range(self.nheads):
                    self.add_module('attention_{}_{}'.format(i + 1, j),
                                    GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=True))

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        input = x
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        if self.nlayers > 2:
            for i in range(self.nlayers - 2):
                temp = []
                x = F.dropout(x, self.dropout, training=self.training)
                cur_input = x
                for j in range(self.nheads):
                    temp.append(self.__getattr__('attention_{}_{}'.format(i + 1, j))(x, adj))
                x = torch.cat(temp, dim=2) + cur_input
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x + input


class TaskSharedEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.__args = args

        # Initialize an LSTM Encoder object.
        self.__encoder = BiLSTMEncoder(
            self.__args.word_embedding_dim,
            self.__args.encoder_hidden_dim,
            self.__args.dropout_rate
        )

        # Initialize an self-attention layer.
        self.__attention = SelfAttention(
            self.__args.word_embedding_dim,
            self.__args.attention_hidden_dim,
            self.__args.attention_output_dim,
            self.__args.dropout_rate
        )

    def forward(self, word_tensor, seq_lens):
        attention_hiddens = self.__attention(word_tensor, seq_lens)
        lstm_hiddens = self.__encoder(word_tensor, seq_lens)
        hiddens = torch.cat([attention_hiddens, lstm_hiddens], dim=2)
        return hiddens


class ModelManager(nn.Module):

    def __init__(self, args, num_word, num_slot, num_intent):
        super(ModelManager, self).__init__()

        self.__num_word = num_word
        self.__num_slot = num_slot
        self.__num_intent = num_intent
        self.__args = args

        # word embedding
        self.__embedding = nn.Embedding(self.__num_word, self.__args.word_embedding_dim)
        # task-shared: self-attentive encoder
        self.__text_encoder = TaskSharedEncoder(args)

        # task-specific encoder
        self.__pre_slot_lstm = BiLSTMEncoder(
            self.__args.encoder_hidden_dim + self.__args.attention_output_dim,
            self.__args.slot_decoder_hidden_dim,
            self.__args.dropout_rate
        )
        self.__intent_lstm = LSTMEncoder(
            self.__args.encoder_hidden_dim + self.__args.attention_output_dim + self.__num_slot,
            self.__args.encoder_hidden_dim + self.__args.attention_output_dim + self.__num_slot,
            self.__args.dropout_rate
        )
        self.__slot_lstm = BiLSTMEncoder(
            self.__args.encoder_hidden_dim + self.__args.attention_output_dim + num_intent,
            self.__args.slot_decoder_hidden_dim,
            self.__args.dropout_rate
        )

        # intent label embedding
        self.__intent_embedding = nn.Parameter(torch.FloatTensor(self.__num_intent, self.__args.intent_embedding_dim))
        nn.init.normal_(self.__intent_embedding.data)
        # Decoder
        # self.__slot_decoder = nn.Sequential(
        #     nn.Linear(self.__args.encoder_hidden_dim + self.__args.attention_output_dim, self.__args.encoder_hidden_dim + self.__args.attention_output_dim),
        #     nn.Dropout(self.__args.dropout_rate),
        #     nn.LeakyReLU(args.alpha),
        #     nn.Linear(self.__args.encoder_hidden_dim + self.__args.attention_output_dim, self.__num_slot)
        # )
        self.__pre_slot_decoder = PreSlotDecoder(
            args,
            self.__args.encoder_hidden_dim + self.__args.attention_output_dim,
            self.__args.slot_decoder_hidden_dim,
            self.__num_slot, self.__args.dropout_rate)
        self.__intent_decoder = nn.Sequential(
            nn.Linear(self.__args.encoder_hidden_dim + self.__args.attention_output_dim + self.__num_slot,
                      self.__args.encoder_hidden_dim + self.__args.attention_output_dim + self.__num_slot),
            nn.LeakyReLU(args.alpha),
            nn.Linear(self.__args.encoder_hidden_dim + self.__args.attention_output_dim + self.__num_slot, self.__num_intent),
        )
        self.__intent_number_decoder = nn.Sequential(
            nn.Linear(self.__args.encoder_hidden_dim + self.__args.attention_output_dim + self.__num_slot,
                      self.__args.encoder_hidden_dim + self.__args.attention_output_dim + self.__num_slot),
            nn.LeakyReLU(args.alpha),
            nn.Linear(self.__args.encoder_hidden_dim + self.__args.attention_output_dim + self.__num_slot,
                      3),
        )
        self.__slot_decoder = SlotDecoder(
            args,
            self.__args.encoder_hidden_dim + self.__args.attention_output_dim,
            self.__args.slot_decoder_hidden_dim,
            self.__num_slot, self.__args.dropout_rate)

    def show_summary(self):
        """
        print the abstract of the defined model.
        """

        print('Model parameters are listed as follows:\n')

        print('\tnumber of word:                            {};'.format(self.__num_word))
        print('\tnumber of slot:                            {};'.format(self.__num_slot))
        print('\tnumber of intent:						    {};'.format(self.__num_intent))
        print('\tword embedding dimension:				    {};'.format(self.__args.word_embedding_dim))
        print('\tencoder hidden dimension:				    {};'.format(self.__args.encoder_hidden_dim))
        print('\tdimension of intent embedding:		    	{};'.format(self.__args.intent_embedding_dim))
        print('\tdimension of slot decoder hidden:  	    {};'.format(self.__args.slot_decoder_hidden_dim))
        print('\thidden dimension of self-attention:        {};'.format(self.__args.attention_hidden_dim))
        print('\toutput dimension of self-attention:        {};'.format(self.__args.attention_output_dim))

        print('\nEnd of parameters show. Now training begins.\n\n')

    def generate_global_adj_gat(self, seq_len, index, batch, window):
        global_intent_idx = [[] for i in range(batch)]
        global_slot_idx = [[] for i in range(batch)]
        for item in index:
            global_intent_idx[item[0]].append(item[1])

        for i, len in enumerate(seq_len):
            global_slot_idx[i].extend(list(range(self.__num_intent, self.__num_intent + len)))

        adj = torch.cat([torch.eye(self.__num_intent + seq_len[0]).unsqueeze(0) for i in range(batch)])
        for i in range(batch):
            for j in global_intent_idx[i]:
                adj[i, j, global_slot_idx[i]] = 1.
                adj[i, j, global_intent_idx[i]] = 1.
            for j in global_slot_idx[i]:
                adj[i, j, global_intent_idx[i]] = 1.

        for i in range(batch):
            for j in range(self.__num_intent, self.__num_intent + seq_len[i]):
                adj[i, j, max(self.__num_intent, j - window):min(seq_len[i] + self.__num_intent, j + window + 1)] = 1.

        if self.__args.row_normalized:
            adj = normalize_adj(adj)
        if self.__args.gpu:
            adj = adj.cuda()
        return adj

    def generate_slot_adj_gat(self, seq_len, batch, window):
        slot_idx_ = [[] for i in range(batch)]
        adj = torch.cat([torch.eye(seq_len[0]).unsqueeze(0) for i in range(batch)])
        for i in range(batch):
            for j in range(seq_len[i]):
                adj[i, j, max(0, j - window):min(seq_len[i], j + window + 1)] = 1.
        if self.__args.row_normalized:
            adj = normalize_adj(adj)
        if self.__args.gpu:
            adj = adj.cuda()
        return adj

    def forward(self, text, seq_lens, n_predicts=None):
        word_tensor = self.__embedding(text)
        g_hiddens = self.__text_encoder(word_tensor, seq_lens)
        # pre-slot
        pre_slot_lstm_out = self.__pre_slot_lstm(g_hiddens, seq_lens)
        pre_slot_lstm_out, pre_pred_slot = self.__pre_slot_decoder(pre_slot_lstm_out, seq_lens)
        # print(pre_slot_lstm_out.size(), pre_pred_slot.size())

        intent_lstm_out = self.__intent_lstm(torch.cat([g_hiddens, pre_slot_lstm_out], dim=-1), seq_lens)
        intent_lstm_out = F.dropout(intent_lstm_out, p=self.__args.dropout_rate, training=self.training)
        pred_intent = self.__intent_decoder(intent_lstm_out)
        """intent num"""
        intent_num_out = intent_lstm_out.mean(dim=1)
        pred_intent__ = self.__intent_decoder(intent_num_out)
        pred_intent_number = self.__intent_number_decoder(intent_num_out)
        """"""
        seq_lens_tensor = torch.tensor(seq_lens)
        if self.__args.gpu:
            seq_lens_tensor = seq_lens_tensor.cuda()
        intent_index_sum = torch.cat(
            [
                torch.sum(torch.sigmoid(pred_intent[i, 0:seq_lens[i], :]) > self.__args.threshold, dim=0).unsqueeze(0)
                for i in range(len(seq_lens))
            ],
            dim=0
        )

        intent_index = (intent_index_sum > (seq_lens_tensor // 2).unsqueeze(1)).nonzero()

        slot_lstm_out = self.__slot_lstm(torch.cat([g_hiddens, pred_intent], dim=-1), seq_lens)
        global_adj = self.generate_global_adj_gat(seq_lens, intent_index, len(pred_intent),
                                                  self.__args.slot_graph_window)
        slot_adj = self.generate_slot_adj_gat(seq_lens, len(pred_intent), self.__args.slot_graph_window)
        pred_slot = self.__slot_decoder(
            slot_lstm_out, seq_lens,
            global_adj=global_adj,
            slot_adj=slot_adj,
            intent_embedding=self.__intent_embedding
        )
        if n_predicts is None:
            return F.log_softmax(pre_pred_slot, dim=1), F.log_softmax(pred_slot, dim=1), pred_intent__, pred_intent_number
        else:
            _, pre_slot_index = pre_pred_slot.topk(n_predicts, dim=1)
            _, slot_index = pred_slot.topk(n_predicts, dim=1)

            intent_idx_ = [[] for _ in range(len(seq_lens))]  # 创建一个嵌套列表
            pred_intent_number_1 = torch.argmax(pred_intent_number, dim=1)
            for i in range(len(seq_lens)):
                num_intents = pred_intent_number_1[i].max(dim=0).values.item() + 1  # 获取每个句子的意图个数
                _, topk_indices = torch.topk(pred_intent__[i], num_intents, dim=0)  # [seq_lens[i], num_intents]
                topk_indices = topk_indices.cpu().data.numpy().tolist()  # 转换为列表
                for item in topk_indices:
                    intent_idx_[i].extend([item])  # 将item包装成列表

            intent_index = [[i, idx] for i, sublist in enumerate(intent_idx_) for idx in sublist]

            return pre_slot_index.cpu().data.numpy().tolist(), slot_index.cpu().data.numpy().tolist(), intent_index, pred_intent_number.cpu().data.numpy().tolist()


class LSTMEncoder(nn.Module):
    """
    Encoder structure based on bidirectional LSTM.
    """

    def __init__(self, embedding_dim, hidden_dim, dropout_rate):
        super(LSTMEncoder, self).__init__()

        # Parameter recording.
        self.__embedding_dim = embedding_dim
        self.__hidden_dim = hidden_dim
        self.__dropout_rate = dropout_rate

        # Network attributes.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__lstm_layer = nn.LSTM(
            input_size=self.__embedding_dim,
            hidden_size=self.__hidden_dim,
            batch_first=True,
            bidirectional=False,
            dropout=self.__dropout_rate,
            num_layers=1
        )

    def forward(self, embedded_text, seq_lens):
        """ Forward process for LSTM Encoder.

        (batch_size, max_sent_len)
        -> (batch_size, max_sent_len, word_dim)
        -> (batch_size, max_sent_len, hidden_dim)

        :param embedded_text: padded and embedded input text.
        :param seq_lens: is the length of original input text.
        :return: is encoded word hidden vectors.
        """

        # Padded_text should be instance of LongTensor.
        dropout_text = self.__dropout_layer(embedded_text)

        # Pack and Pad process for input of variable length.
        packed_text = pack_padded_sequence(dropout_text, seq_lens, batch_first=True)
        lstm_hiddens, (h_last, c_last) = self.__lstm_layer(packed_text)
        padded_hiddens, _ = pad_packed_sequence(lstm_hiddens, batch_first=True)

        return padded_hiddens


class BiLSTMEncoder(nn.Module):
    """
    Encoder structure based on bidirectional LSTM.
    """

    def __init__(self, embedding_dim, hidden_dim, dropout_rate):
        super(BiLSTMEncoder, self).__init__()

        # Parameter recording.
        self.__embedding_dim = embedding_dim
        self.__hidden_dim = hidden_dim // 2
        self.__dropout_rate = dropout_rate

        # Network attributes.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__lstm_layer = nn.LSTM(
            input_size=self.__embedding_dim,
            hidden_size=self.__hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout=self.__dropout_rate,
            num_layers=1
        )

    def forward(self, embedded_text, seq_lens):
        """ Forward process for LSTM Encoder.

        (batch_size, max_sent_len)
        -> (batch_size, max_sent_len, word_dim)
        -> (batch_size, max_sent_len, hidden_dim)

        :param embedded_text: padded and embedded input text.
        :param seq_lens: is the length of original input text.
        :return: is encoded word hidden vectors.
        """

        # Padded_text should be instance of LongTensor.
        dropout_text = self.__dropout_layer(embedded_text)

        # Pack and Pad process for input of variable length.
        packed_text = pack_padded_sequence(dropout_text, seq_lens, batch_first=True)
        lstm_hiddens, (h_last, c_last) = self.__lstm_layer(packed_text)
        padded_hiddens, _ = pad_packed_sequence(lstm_hiddens, batch_first=True)

        return padded_hiddens


class PreSlotDecoder(nn.Module):
    """
    Decoder structure based on unidirectional LSTM. 其实没有用LSTM解码
    """

    def __init__(self, args, input_dim, hidden_dim, output_dim, embedding_dim=None, extra_dim=None):
        """ Construction function for Decoder.

        :param input_dim: input dimension of Decoder. In fact, it's encoder hidden size.
        :param hidden_dim: hidden dimension of iterative LSTM.
        :param output_dim: output dimension of Decoder. In fact, it's total number of intent or slot.
        :param dropout_rate: dropout rate of network which is only useful for embedding.
        """

        super(PreSlotDecoder, self).__init__()
        self.__args = args
        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        # self.__dropout_rate = dropout_rate
        self.__embedding_dim = embedding_dim
        self.__extra_dim = extra_dim

        self.__linear_layer = nn.Sequential(
            nn.Linear(self.__hidden_dim,
                      self.__hidden_dim),
            nn.LeakyReLU(args.alpha),
            nn.Linear(self.__hidden_dim, self.__output_dim),
        )

    def forward(self, encoded_hiddens, seq_lens, global_adj=None, slot_adj=None, intent_embedding=None):
        """ Forward process for decoder.

        :param encoded_hiddens: is encoded hidden tensors produced by encoder.
        :param seq_lens: is a list containing lengths of sentence.
        :return: is distribution of prediction labels.
        """

        output_tensor_list, sent_start_pos = [], 0

        out = self.__linear_layer(encoded_hiddens)  # torch.Size([16, 53, 117])

        for i in range(0, len(seq_lens)):
            output_tensor_list.append(out[i, 0:0 + seq_lens[i]])  # num_intent:num_intent

        a = torch.cat(output_tensor_list, dim=0)
        return out, a


class SlotDecoder(nn.Module):
    """
    Decoder structure based on unidirectional LSTM.
    """

    def __init__(self, args, input_dim, hidden_dim, output_dim, dropout_rate, embedding_dim=None, extra_dim=None):
        """ Construction function for Decoder.

        :param input_dim: input dimension of Decoder. In fact, it's encoder hidden size.
        :param hidden_dim: hidden dimension of iterative LSTM.
        :param output_dim: output dimension of Decoder. In fact, it's total number of intent or slot.
        :param dropout_rate: dropout rate of network which is only useful for embedding.
        """

        super(SlotDecoder, self).__init__()
        self.__args = args
        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate
        self.__embedding_dim = embedding_dim
        self.__extra_dim = extra_dim

        # If embedding_dim is not None, the output and input
        # of this structure is relevant.
        if self.__embedding_dim is not None:
            self.__embedding_layer = nn.Embedding(output_dim, embedding_dim)
            self.__init_tensor = nn.Parameter(
                torch.randn(1, self.__embedding_dim),
                requires_grad=True
            )

        # Network parameter definition.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)

        self.__slot_graph = GAT(
            self.__hidden_dim,
            self.__args.decoder_gat_hidden_dim,
            self.__hidden_dim,
            self.__args.gat_dropout_rate, self.__args.alpha, self.__args.n_heads,
            self.__args.n_layers_decoder_global)

        self.__global_graph = GAT(
            self.__hidden_dim,
            self.__args.decoder_gat_hidden_dim,
            self.__hidden_dim,
            self.__args.gat_dropout_rate, self.__args.alpha, self.__args.n_heads,
            self.__args.n_layers_decoder_global)

        self.__linear_layer = nn.Sequential(
            nn.Linear(self.__hidden_dim,
                      self.__hidden_dim),
            nn.LeakyReLU(args.alpha),
            nn.Linear(self.__hidden_dim, self.__output_dim),
        )

    def forward(self, encoded_hiddens, seq_lens, global_adj=None, slot_adj=None, intent_embedding=None):
        """ Forward process for decoder.

        :param encoded_hiddens: is encoded hidden tensors produced by encoder.
        :param seq_lens: is a list containing lengths of sentence.
        :return: is distribution of prediction labels.
        """

        input_tensor = encoded_hiddens
        output_tensor_list, sent_start_pos = [], 0

        batch = len(seq_lens)
        slot_graph_out = self.__slot_graph(encoded_hiddens, slot_adj)
        intent_in = intent_embedding.unsqueeze(0).repeat(batch, 1, 1)
        global_graph_in = torch.cat([intent_in, slot_graph_out], dim=1)
        global_graph_out = self.__global_graph(global_graph_in, global_adj)
        out = self.__linear_layer(global_graph_out)
        num_intent = intent_embedding.size(0)
        for i in range(0, len(seq_lens)):
            output_tensor_list.append(out[i, num_intent:num_intent + seq_lens[i]])
        return torch.cat(output_tensor_list, dim=0)


class QKVAttention(nn.Module):
    """
    Attention mechanism based on Query-Key-Value architecture. And
    especially, when query == key == value, it's self-attention.
    """

    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, output_dim, dropout_rate):
        super(QKVAttention, self).__init__()

        # Record hyper-parameters.
        self.__query_dim = query_dim
        self.__key_dim = key_dim
        self.__value_dim = value_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        # Declare network structures.
        self.__query_layer = nn.Linear(self.__query_dim, self.__hidden_dim)
        self.__key_layer = nn.Linear(self.__key_dim, self.__hidden_dim)
        self.__value_layer = nn.Linear(self.__value_dim, self.__output_dim)
        self.__dropout_layer = nn.Dropout(p=self.__dropout_rate)

    def forward(self, input_query, input_key, input_value):
        """ The forward propagation of attention.

        Here we require the first dimension of input key
        and value are equal.

        :param input_query: is query tensor, (n, d_q)
        :param input_key:  is key tensor, (m, d_k)
        :param input_value:  is value tensor, (m, d_v)
        :return: attention based tensor, (n, d_h)
        """

        # Linear transform to fine-tune dimension.
        linear_query = self.__query_layer(input_query)
        linear_key = self.__key_layer(input_key)
        linear_value = self.__value_layer(input_value)

        score_tensor = F.softmax(torch.matmul(
            linear_query,
            linear_key.transpose(-2, -1)
        ), dim=-1) / math.sqrt(self.__hidden_dim)
        forced_tensor = torch.matmul(score_tensor, linear_value)
        forced_tensor = self.__dropout_layer(forced_tensor)

        return forced_tensor


class SelfAttention(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(SelfAttention, self).__init__()

        # Record parameters.
        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        # Record network parameters.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__attention_layer = QKVAttention(
            self.__input_dim, self.__input_dim, self.__input_dim,
            self.__hidden_dim, self.__output_dim, self.__dropout_rate
        )

    def forward(self, input_x, seq_lens):
        dropout_x = self.__dropout_layer(input_x)
        attention_x = self.__attention_layer(
            dropout_x, dropout_x, dropout_x
        )

        return attention_x
