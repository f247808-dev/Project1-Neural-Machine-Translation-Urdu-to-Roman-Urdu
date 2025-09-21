# app.py
import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# -------------------------------
# 1️⃣ Define model classes
# -------------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg=None, teacher_forcing_ratio=0.0):
        # Greedy decoding if trg=None
        batch_size = src.shape[0]
        max_len = trg.shape[1] if trg is not None else 50
        trg_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, max_len, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        input = torch.tensor([SOS_token]*batch_size).to(self.device)

        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input = top1

        return outputs

# -------------------------------
# 2️⃣ Load vocab and tokens
# -------------------------------
# Replace with your actual vocab dictionaries
# Urdu -> index
source_vocab = {'ا':0, 'ب':1, 'پ':2}  # etc.
# Roman Urdu -> index
target_vocab = {'a':0, 'b':1, 'p':2, '<SOS>':3, '<EOS>':4}  # etc.
target_vocab_inv = {v:k for k,v in target_vocab.items()}

SOS_token = target_vocab['<SOS>']
EOS_token = target_vocab['<EOS>']

# -------------------------------
# 3️⃣ Load trained model
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_model(path):
    # Define encoder and decoder with correct dimensions
    INPUT_DIM = len(source_vocab)
    OUTPUT_DIM = len(target_vocab)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    DROPOUT = 0.5

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)
    
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# Load your trained model here
model = load_model("your_model.pt")  # replace with actual path

# -------------------------------
# 4️⃣ Tokenization functions
# -------------------------------
def tokenize(text, vocab):
    return [vocab[char] for char in text if char in vocab]

def detokenize(indices, vocab_inv):
    # stop at <EOS> if exists
    result = []
    for idx in indices:
        if idx == EOS_token:
            break
        result.append(vocab_inv.get(idx, ''))
    return ''.join(result)

# -------------------------------
# 5️⃣ Streamlit UI
# -------------------------------
st.title("Urdu → Roman Urdu Translator")
st.write("Enter Urdu text below and get Roman Urdu translation:")

input_text = st.text_area("Urdu Text:")

if st.button("Translate"):
    if not input_text.strip():
        st.warning("Please enter some text.")
    else:
        st.info("Translating...")
        # Tokenize
        tokenized_input = tokenize(input_text, source_vocab)
        src_tensor = torch.tensor([tokenized_input]).to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(src_tensor)
            predicted_ids = outputs.argmax(2).squeeze().tolist()
        
        translation = detokenize(predicted_ids, target_vocab_inv)
        st.success(translation)
