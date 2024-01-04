from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import torch

st.title('Speech Generator')


synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

text = st.text_input("Enter text up to 500 words")
if st.button('Generate Speech'):
    speech = synthesiser(text, forward_params={"speaker_embeddings": speaker_embedding})
    # st.write(speech)
    st.audio(speech['audio'], sample_rate = speech['sampling_rate'])
