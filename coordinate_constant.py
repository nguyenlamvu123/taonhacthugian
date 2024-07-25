from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch, json, os, scipy, time
from functools import wraps
import numpy as np
from audiocraft.data.audio import audio_write


def timer(func):  # @timer
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        if debug:
            print(f"Execution time of {func.__name__}: {end - start} seconds")
        return result
    return wrapper


def readfile(file="uid.txt", mod="r", cont=None, jso: bool = False):
    if not mod in ("w", "a", ):
        assert os.path.isfile(file), str(file)
    if mod == "r":
        with open(file, encoding="utf-8") as file:
            lines: list = file.readlines()
        return lines
    elif mod == "_r":
        with open(file, encoding="utf-8") as file:
            contents = file.read() if not jso else json.load(file)
        return contents
    elif mod == "rb":
        with open(file, mod) as file:
            contents = file.read()
        return contents
    elif mod in ("w", "a", ):
        with open(file, mod, encoding="utf-8") as fil_e:
            if not jso:
                fil_e.write(cont)
            else:
                json.dump(cont, fil_e, indent=2, ensure_ascii=False)


def sample_length2num_tokens(sample_length=30):
    "sample_length: seconds"
    return sample_length * model.config.audio_encoder.frame_rate + 3


def apply_nltk(func):  # @apply_nltk
    def gener_dat(audio_values, met):
        for dat in audio_values:
            for dat_ in dat:
                    yield dat_.cpu().numpy()

    @wraps(func)
    def wrapper(*args, **kwargs):
        silence = np.zeros(int(0.25 * sampling_rate))
        assert 'thoigian' in kwargs
        thoigian = kwargs['thoigian']
        outlocat = kwargs['outlocat']

        pieces = []
        while thoigian > 0:
            thoigian_ = thoigian - sample_length
            if thoigian_ < 0:
                kwargs['thoigian'] = thoigian
                audio_values = func(**kwargs)
            else:
                kwargs['thoigian'] = sample_length
                audio_values = func(**kwargs)
            thoigian = thoigian_

            sil = silence.copy()
            if len(pieces) == 0:
                for dat_ in gener_dat(audio_values, kwargs['met']):
                    pieces.append([dat_, sil])
            else:
                dat_ = audio_values[0][0].cpu().numpy()
                # for dat_ in gener_dat(audio_values, kwargs['met'])):  # TODO mutipl origin pieces list
                for piece in pieces:
                    piece += [dat_, sil]
        listfilenames = list()
        for enu, piece in enumerate(pieces):
            out___mp4_ = os.path.join(os.path.dirname(__file__), f"musicgen_out_{enu}.wav") if outlocat is None \
                else f"{os.path.splitext(outlocat)[0]}_{enu}.wav"
            data = np.concatenate(piece)
            scipy.io.wavfile.write(out___mp4_, rate=sampling_rate, data=data)
            listfilenames.append((out___mp4_, data))
        return listfilenames
    return wrapper

pret_loca = os.path.join(os.path.dirname(__file__), 'pret')
device = "cuda:0" if torch.cuda.is_available() else "cpu"
sample_length = 30  # seconds
g_scale = 5
historyfile: str = "hist.txt"
debug = False

# site: https://huggingface.co/docs/transformers/main/en/model_doc/musicgen
pretra: tuple = (
    "facebook/musicgen-small",  # 300M model, text to music only - 🤗 Hub
    "facebook/musicgen-medium",  # 1.5B model, text to music only - 🤗 Hub
    "facebook/musicgen-melody",  # 1.5B model, text to music and text+melody to music - 🤗 Hub
    "facebook/musicgen-large",  # 3.3B model, text to music only - 🤗 Hub
    "facebook/musicgen-melody-large",  # 3.3B model, text to music and text+melody to music - 🤗 Hub
    "facebook/musicgen-stereo-*",  # or all in one
)

def aucr_model(pre='melody'):
    from audiocraft.models import MusicGen
    return MusicGen.get_pretrained(pre, cache_dir=pret_loca)


processor = AutoProcessor.from_pretrained(pretra[0], cache_dir=pret_loca)
model = MusicgenForConditionalGeneration.from_pretrained(pretra[0], cache_dir=pret_loca)
model = model.to(device)
au_crmode = aucr_model()
# model = configgg(model)
sampling_rate = model.config.audio_encoder.sampling_rate

# num_tokens = 1503  # 256
num_tokens = sample_length2num_tokens()
# https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/250-music-generation/250-music-generation.ipynb#scrollTo=ae8f6270-e745-4adb-b65d-c1d8dc44d7fc
