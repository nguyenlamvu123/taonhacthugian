from transformers import AutoProcessor, MusicgenForConditionalGeneration
import os, torch, scipy
from functools import wraps
import numpy as np


def sample_length2num_tokens(sample_length=30):
    "sample_length: seconds"
    return sample_length * model.config.audio_encoder.frame_rate + 3

pret_loca = os.path.join(os.path.dirname(__file__), 'pret')
device = "cuda:0" if torch.cuda.is_available() else "cpu"
sample_length = 30  # seconds
g_scale = 5
debug = True

# site: https://huggingface.co/docs/transformers/main/en/model_doc/musicgen
pretra: tuple = (
    "facebook/musicgen-small",  # 300M model, text to music only - 🤗 Hub
    "facebook/musicgen-medium",  # 1.5B model, text to music only - 🤗 Hub
    "facebook/musicgen-melody",  # 1.5B model, text to music and text+melody to music - 🤗 Hub
    "facebook/musicgen-large",  # 3.3B model, text to music only - 🤗 Hub
    "facebook/musicgen-melody-large",  # 3.3B model, text to music and text+melody to music - 🤗 Hub
    "facebook/musicgen-stereo-*:",  # or all in one
)
processor = AutoProcessor.from_pretrained(pretra[0], cache_dir=pret_loca)
model = MusicgenForConditionalGeneration.from_pretrained(pretra[0], cache_dir=pret_loca)
model = model.to(device)
# model = configgg(model)
sampling_rate = model.config.audio_encoder.sampling_rate

# num_tokens = 1503  # 256
num_tokens = sample_length2num_tokens()
# https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/250-music-generation/250-music-generation.ipynb#scrollTo=ae8f6270-e745-4adb-b65d-c1d8dc44d7fc
