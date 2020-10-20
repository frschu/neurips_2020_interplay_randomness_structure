import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, rnn_type, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx, train_embedding=True):
        super().__init__()
        self.rnn_type = rnn_type
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        
        # Cell states
        if rnn_type == 'RNN':
            self.keys_rnn = ["c"]
        elif rnn_type == 'GRU':
            self.keys_rnn = ["r", "z", "c"] # c = n, but we need this label.
        elif rnn_type == 'LSTM':
            self.keys_rnn = ["i", "f", "c", "o"] # c = g.
        else:
            raise NotImplementedError()
        # Number of states
        self.n_states = len(self.keys_rnn)

        # Number of directions
        n_directions = 1 + int(bidirectional)
        self.n_directions = n_directions
        # Total number of distinguishable weights
        self.n_rnn = self.n_states * n_directions
        
        # Define layers
        # Embedding. Set padding_idx so that the <pad> is not updated. 
        self.encoder = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        # RNN
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=n_layers,
                              bidirectional=bidirectional, dropout=dropout)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                               bidirectional=bidirectional, dropout=dropout)
        # Decoder: fully-connected
        self.decoder = nn.Linear(hidden_dim * n_directions, output_dim)
    
        # Train the embedding?
        if not train_embedding:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Setup dropout
        self.drop = nn.Dropout(dropout)
        
        
    def forward(self, batch_text):
        text, text_lengths = batch_text
        #text = [sentence len, batch size]
        embedded = self.encoder(text)
        #embedded = [sent len, batch size, emb dim]
        
        # Pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        if self.rnn_type == 'LSTM':
            packed_output, (hidden, cell) = self.rnn(packed_embedded)
        else:
            packed_output, hidden = self.rnn(packed_embedded)
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]
        # Unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        #output = [sentence len, batch size, hid dim * num directions]
        
        # Concatenate the final hidden layers (for bidirectional)
        # hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        hidden = (torch.transpose(hidden[-self.n_directions:], 0, 1)).reshape(
            (-1, self.hidden_dim * self.n_directions))
        
        # Dropout
        hidden = self.drop(hidden)
        
        # Decode
        decoded = self.decoder(hidden).squeeze(1)
        
        return decoded