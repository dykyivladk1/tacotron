import torch
import torch.nn as nn

class Prenet(nn.Module):
    
    def __init__(self, in_dim, hidden_sizes=[256, 128]):
        super(Prenet, self).__init__()
        self.layers = nn.ModuleList()
        
        previous_size = in_dim
        for hidden_size in hidden_sizes:
            layer = nn.Linear(previous_size, hidden_size)
            self.layers.append(layer)
            previous_size = hidden_size
        
        self.activation = nn.ReLU()
        self.dropout_rate = 0.5

    def forward(self, x):
        
        for layer in self.layers:
            x = self.activation(layer(x))
            x = nn.functional.dropout(x, p=self.dropout_rate, training=self.training)
        return x

class BatchNormConv1d(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, stride, padding, activation=None):
        super(BatchNormConv1d, self).__init__()
        self.conv1d = nn.Conv1d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_size, momentum=0.99, eps=1e-3)
        self.activation = activation

    def forward(self, x):
        x = self.conv1d(x)
        if self.activation is not None:
            x = self.activation(x)
        x = self.bn(x)
        return x

class Highway(nn.Module):
    def __init__(self, in_size, out_size):
        super(Highway, self).__init__()
        self.H = nn.Linear(in_size, out_size)
        self.H.bias.data.zero_()
        self.T = nn.Linear(in_size, out_size)
        self.T.bias.data.fill_(-1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        H = self.relu(self.H(x))
        T = self.sigmoid(self.T(x))
        C = 1.0 - T
        y = H * T + x * C
        return y

class CBHG(nn.Module):
    
    def __init__(self, in_dim, K=16, hidden_sizes=[128, 128]):
        super(CBHG, self).__init__()
        self.in_dim = in_dim
        self.relu = nn.ReLU()
        self.conv1d_banks = nn.ModuleList(
            [BatchNormConv1d(in_dim, in_dim, kernel_size=k, stride=1,
                             padding=k // 2, activation=self.relu)
             for k in range(1, K + 1)])
        self.pool1d = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        in_sizes = [K * in_dim] + hidden_sizes[:-1]
        activations = [self.relu] * (len(hidden_sizes) - 1) + [None]
        self.conv1d_projs = nn.ModuleList(
            [BatchNormConv1d(in_size, out_size, kernel_size=3,
                             stride=1, padding=1, activation=act)
             for in_size, out_size, act in zip(in_sizes, hidden_sizes, activations)])

        self.pre_highway_proj = nn.Linear(hidden_sizes[-1], in_dim, bias=False)
        self.highways = nn.ModuleList(
            [Highway(in_dim, in_dim) for _ in range(4)])
        self.gru = nn.GRU(
            in_dim, in_dim, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, inputs, input_lengths=None):
        x = inputs
        
        assert x.size(-1) == self.in_dim
        
        x = x.transpose(1, 2)
        T = x.size(-1)

        conv_outputs = []
        for conv1d in self.conv1d_banks:
            conv_output = conv1d(x)
            conv_outputs.append(conv_output[:, :, :T])
        x = torch.cat(conv_outputs, dim=1)

        assert x.size(1) == self.in_dim * len(self.conv1d_banks)
        x = self.pool1d(x)[:, :, :T]

        for conv1d in self.conv1d_projs:
            x = conv1d(x)

        x = x.transpose(1, 2)
        
        x = self.pre_highway_proj(x)

        x += inputs
        for highway in self.highways:
            x = highway(x)

        if input_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, input_lengths, batch_first=True, enforce_sorted=False)
        
        y, _ = self.gru(x)

        if input_lengths is not None:
            y, _ = nn.utils.rnn.pad_packed_sequence(
                y, batch_first=True)
        return y

class Encoder(nn.Module):
    
    def __init__(self, in_dim):
        super(Encoder, self).__init__()
        self.prenet = Prenet(in_dim, hidden_sizes=[256, 128])
        self.cbhg = CBHG(128, K=16, hidden_sizes=[128, 128])

    def forward(self, x, input_lengths=None):
        x = self.prenet(x)
        x = self.cbhg(x, input_lengths)
        return x

class BahdanauAttn(nn.Module):
    
    def __init__(self, size):
        super(BahdanauAttn, self).__init__()
        self.query_layer = nn.Linear(size, size, bias=False)
        self.tanh = nn.Tanh()
        self.v = nn.Linear(size, 1, bias=False)

    def forward(self, query, memory):
        
        if query.dim() == 2:
            
            query = query.unsqueeze(1)

        Q = self.query_layer(query)
        K = memory
        
        alignment = self.v(self.tanh(Q + K))
        
        alignment = alignment.squeeze(-1)
        return alignment

class AttnWrapper(nn.Module):
    def __init__(self, rnn_cell, attn_mechanism):
        super(AttnWrapper, self).__init__()
        self.rnn_cell = rnn_cell
        self.attn_mechanism = attn_mechanism
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, prev_ctx, cell_state, memory, processed_mem):
        
        cell_input = torch.cat([query, prev_ctx], dim=-1)
        
        cell_output = self.rnn_cell(cell_input, cell_state)
        
        alignment_scores = self.attn_mechanism(cell_output, processed_mem)
        
        attention_weights = self.softmax(alignment_scores)
        
        context_vector = torch.bmm(attention_weights.unsqueeze(1), memory)
        
        context_vector = context_vector.squeeze(1)
        
        return cell_output, context_vector, attention_weights

class MelDecoder(nn.Module):
    
    def __init__(self, in_size, r):
        super(MelDecoder, self).__init__()
        self.in_size = in_size
        self.r = r
        self.prenet = Prenet(in_size * r, hidden_sizes=[256, 128])
        
        self.attn_rnn = AttnWrapper(
            nn.GRUCell(256 + 128, 256),
            BahdanauAttn(256))
        self.memory_layer = nn.Linear(256, 256, bias=False)
        
        self.pre_rnn_dec_proj = nn.Linear(512, 256)
        self.rnns_dec = nn.ModuleList(
                [nn.GRUCell(256, 256) for _ in range(2)])
        self.mel_proj = nn.Linear(256, in_size * r)
        self.max_decode_steps = 200

    def forward(self, encoder_outputs, inputs=None):
        
        batch_size = encoder_outputs.size(0)
        processed_mem = self.memory_layer(encoder_outputs)
        
        greedy = inputs is None

        if not greedy:
            inputs = inputs.view(batch_size, inputs.size(1) // self.r, -1)
            inputs = inputs.transpose(0, 1)
            T_dec = inputs.size(0)

        init_input = torch.zeros(batch_size, self.in_size * self.r, device=encoder_outputs.device)
        
        attn_rnn_hidden = torch.zeros(batch_size, 256, device=encoder_outputs.device)
        
        rnns_dec_hidden = [torch.zeros(batch_size, 256, device=encoder_outputs.device)
                for _ in range(len(self.rnns_dec))]
        
        curr_ctx = torch.zeros(batch_size, 256, device=encoder_outputs.device)

        outputs = []
        alignments = []

        t = 0
        while True:
            if t == 0:
                curr_input = init_input
            else:
                curr_input = outputs[-1] if greedy else inputs[t-1]

            curr_input = self.prenet(curr_input)
            attn_rnn_hidden, curr_ctx, alignment = self.attn_rnn(
                    curr_input, curr_ctx, attn_rnn_hidden, encoder_outputs, processed_mem)

            decoder_input = self.pre_rnn_dec_proj(
                    torch.cat([attn_rnn_hidden, curr_ctx], dim=-1))
            
            for i in range(len(self.rnns_dec)):
                rnns_dec_hidden[i] = self.rnns_dec[i](decoder_input, rnns_dec_hidden[i])
                
                decoder_input = rnns_dec_hidden[i] + decoder_input

            output = self.mel_proj(decoder_input)
            outputs.append(output)
            alignments.append(alignment)
            t += 1

            if greedy:
                if (t > 1 and self.is_eof(output)) or (t > self.max_decode_steps):
                    break
            else:
                if t >= T_dec:
                    break

        assert greedy or len(outputs) == T_dec
        
        alignments = torch.stack(alignments).transpose(0, 1)
        outputs = torch.stack(outputs).transpose(0, 1).contiguous()
        return outputs, alignments

    def is_eof(self, output, eps=0.2):
        
        return (output <= eps).all()

class Tacotron(nn.Module):
    def __init__(self, n_vocab, embedding_size=256, mel_size=80, linear_size=1025, r=5):
        super(Tacotron, self).__init__()
        self.mel_size = mel_size
        self.linear_size = linear_size
        self.embedding = nn.Embedding(n_vocab, embedding_size)
        
        self.embedding.weight.data.normal_(0, 0.3)
        self.encoder = Encoder(embedding_size)
        self.mel_decoder = MelDecoder(mel_size, r)
        self.postnet = CBHG(mel_size, K=8, hidden_sizes=[256, mel_size])
        self.last_proj = nn.Linear(mel_size * 2, linear_size)

    def forward(self, texts, melspec=None, text_lengths=None):
        batch_size = texts.size(0)
        txt_feat = self.embedding(texts)
        
        encoder_outputs = self.encoder(txt_feat, text_lengths)
        mel_outputs, alignments = self.mel_decoder(encoder_outputs, melspec)
        
        mel_outputs = mel_outputs.view(batch_size, -1, self.mel_size)
        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_proj(linear_outputs)
        return mel_outputs, linear_outputs, alignments
