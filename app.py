# app.py
import sys
import os
import torch
import torch.nn as nn
import streamlit as st

# -------------------------------
# Ensure neural_machine_training.py is found
# -------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from neural_machine_training import Encoder, Decoder, Seq2Seq
except ModuleNotFoundError:
    st.error("neural_machine_training.py not found! Make sure it is in the same folder as app.py")
    st.stop()

# -------------------------------
# Device configuration
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")

# -------------------------------
# Model and hyperparameters
# -------------------------------
INPUT_DIM = 1000      # Example: size of Urdu vocab
OUTPUT_DIM = 1000     # Example: size of Roman Urdu vocab
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

# Initialize encoder, decoder, and seq2seq
encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
model = Seq2Seq(encoder, decoder, device).to(device)

# -------------------------------
# Load trained model weights
# -------------------------------
checkpoint_path = os.path.join(current_dir, "checkpoints", "seq2seq_model.pt")
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    st.success("Model loaded successfully!")
else:
    st.warning(f"Checkpoint not found at {checkpoint_path}. The app will not work without a trained model.")

# -------------------------------
# Example vocab dictionaries
# -------------------------------
# Replace with your actual vocab mappings
src_vocab = {'<unk>':0, 'میرا':1, 'نام':2, 'علی':3}
trg_vocab = {0:'<unk>', 1:'Mera', 2:'naam', 3:'Ali'}

# -------------------------------
# Translation function
# -------------------------------
def translate(sentence, src_vocab, trg_vocab, max_len=50):
    """
    sentence: list of Urdu tokens
    src_vocab, trg_vocab: dicts {word:index} and {index:word}
    """
    model.eval()
    
    src_indexes = [src_vocab.get(token, src_vocab['<unk>']) for token in sentence]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)  # [seq_len, 1]
    
    with torch.no_grad():
        outputs = model(src_tensor, trg=None, teacher_forcing_ratio=0)  # no teacher forcing
    
    trg_indexes = outputs.argmax(2).squeeze(1).tolist()
    trg_tokens = [trg_vocab.get(idx, '<unk>') for idx in trg_indexes]
    
    return ' '.join(trg_tokens)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Urdu → Roman Urdu Translator")
st.write("Type your Urdu sentence below:")

user_input = st.text_area("Enter Urdu text here:")

if st.button("Translate"):
    if not user_input.strip():
        st.warning("Please enter a sentence to translate.")
    else:
        # Simple whitespace tokenizer, replace with your tokenizer if needed
        tokens = user_input.strip().split()
        translation = translate(tokens, src_vocab, trg_vocab)
        st.subheader("Roman Urdu Translation:")
        st.write(translation)
