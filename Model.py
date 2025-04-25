# Model.py

import torch
import torch.nn as nn

class classifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, bidirectional, dropout):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           dropout=dropout if num_layers > 1 else 0,
                           batch_first=True) # batch_first=True is important

        # The number of directions in the final output from the RNN
        # is 2 if bidirectional, 1 otherwise.
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        # text = [batch size, seq len]

        embedded = self.dropout(self.embedding(text))
        # embedded = [batch size, seq len, embedding dim]

        # pack sequence
        # text_lengths must be on CPU for pack_padded_sequence
        # and must be a LongTensor
        # CORRECTED LINE: Added enforce_sorted=False
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded,
            text_lengths.cpu(), # Ensure lengths are on CPU
            batch_first=True,
            enforce_sorted=False # This line fixes the RuntimeError
        )

        # Pass packed sequence through RNN
        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # unpack sequence (if needed for future layers, but not for final classification)
        # output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # output = [batch size, seq len, hidden dim * num directions]

        # hidden = [num layers * num directions, batch size, hidden dim]
        # cell = [num layers * num directions, batch size, hidden dim]

        # We take the final hidden state(s) for classification
        # If bidirectional, concatenate the final forward and backward hidden states
        if self.rnn.bidirectional:
            # Concatenate the last forward (hidden[-2, :, :]) and last backward (hidden[-1, :, :]) hidden states
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            # Take the last hidden state for unidirectional RNN
            hidden = self.dropout(hidden[-1, :, :])

        # hidden = [batch size, hidden dim * num directions]

        # Pass through linear layer to get logits
        # The output of the linear layer will be [batch size, output_dim]
        # CrossEntropyLoss expects this shape.
        return self.fc(hidden)