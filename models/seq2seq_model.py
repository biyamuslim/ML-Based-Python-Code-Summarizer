import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        # Turn token IDs into vectors before passing them through the LSTM.
        embedded = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell

# helps decoder decide which parts of the input code to focus on when generating each word in the summary. 
# Decoder uses attention weights to build a context vector.
class Attention(nn.Module): # nn.Module : Pytorch module for attention.
    def __init__(self, hidden_dim): # initializes the attention layer
        super().__init__()
        self.energy = nn.Linear(hidden_dim * 2, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False) # convert combine vector to one score
        #score tells how important is the input token right now for generating the next summary token. Higher score = more important.

    # hidden = is the decoder’s current hidden state.
    # encoder_outputs = the outputs from the encoder for all input tokens.
    def forward(self, hidden, encoder_outputs): 
        src_len = encoder_outputs.shape[1] #encoder has one output for each token.

        # Compare the current decoder state with every encoder output.
        hidden = hidden[-1].unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.energy(torch.cat((hidden, encoder_outputs), dim=2))) #How well does the decoder state match this input token?
        attention = self.score(energy).squeeze(2) #converts each token’s energy into one number.

        # Softmax converts attention scores for all input tokens into attention weights.
        return torch.softmax(attention, dim=1)

#generates the summary one token at a time.
class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, dropout=0.0):
        super().__init__()
        #converts the output token ID into a vector,
        self.embedding = nn.Embedding(output_dim, embed_dim) # {token ID for "return" → embedding vector}
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(hidden_dim)
        # decoder LSTM input is embedded previous token + attention context
        self.lstm = nn.LSTM(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        # predicts next token 
        self.fc = nn.Linear(embed_dim + hidden_dim * 2, output_dim)

    def forward(self, x, hidden, cell, encoder_outputs):
        x = x.unsqueeze(1)
        embedded = self.dropout(self.embedding(x))

        # Use attention to build a context vector from the encoder outputs.
        attention = self.attention(hidden, encoder_outputs).unsqueeze(1)
        context = torch.bmm(attention, encoder_outputs)
        lstm_input = torch.cat((embedded, context), dim=2)

        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction_features = self.dropout(torch.cat(
            (output.squeeze(1), context.squeeze(1), embedded.squeeze(1)),
            dim=1
        ))
        prediction = self.fc(prediction_features)

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # Encode the whole input code sequence once.
        encoder_outputs, hidden, cell = self.encoder(src)
        input = trg[:, 0]

        # Decode the summary one token at a time.
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t] = output

            # Teacher forcing sometimes feeds the real previous token during training.
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)

            input = trg[:, t] if teacher_force else top1

        return outputs
