import streamlit as st
import torch
from neural_machine_training import Encoder, Decoder, Seq2Seq

# ------------------------------
# Load model
# ------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 5000
OUTPUT_DIM = 5000

encoder = Encoder(INPUT_DIM).to(DEVICE)
decoder = Decoder(OUTPUT_DIM).to(DEVICE)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

checkpoint_path = "checkpoints/seq2seq_epoch5.pt"
try:
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    st.success("✅ Model loaded successfully")
except:
    st.warning("⚠️ No trained model found, using random weights")

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("Urdu → Roman Urdu Transliteration")
st.write("Enter Urdu text below:")

urdu_text = st.text_input("Input Urdu text:")
if st.button("Transliterate"):
    if urdu_text.strip() == "":
        st.error("Please enter some Urdu text")
    else:
        # TODO: Replace with tokenizer + model prediction
        roman_text = "demo_output"  # placeholder
        st.success(f"Roman Urdu: {roman_text}")
