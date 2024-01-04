import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

st.title('Speech Generator')
st.text_input

embeddings_dataset = st.session_state.cache['EMBED']
synthesiser = st.session_state.cache['AUDIO_PIPE']
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    
speech = synthesiser(text, forward_params={"speaker_embeddings": speaker_embedding})
# st.write(speech)
st.audio(speech['audio'], sample_rate = speech['sampling_rate'])
