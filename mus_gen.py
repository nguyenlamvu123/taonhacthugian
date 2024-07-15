from transformers import AutoProcessor, MusicgenForConditionalGeneration
import os, scipy

pret_loca = os.path.join(os.path.dirname(__file__), 'pret')
num_tokens = 1503  # 256

def configgg(model):
    model.generation_config.guidance_scale = 4.0
    model.generation_config.max_length = 256
    return model


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
# model = configgg(model)

unconditional_inputs = model.get_unconditional_inputs(num_samples=1)
audio_values = model.generate(**unconditional_inputs, do_sample=True, max_new_tokens=256)
for input_text in (
        (["epic movie theme", "sad jazz", ], "emtsj.wav"),
        (["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums", ], "80spt_90srs.wav"),
        (["80s blues track with groovy saxophone"], "80sbt.wav"),
):
    inputs = processor(
        text=input_text[0],
        padding=True,
        return_tensors="pt",
    )
    
    audio_values = model.generate(
        **inputs,
        max_new_tokens=num_tokens,
        guidance_scale=3,  # The guidance_scale is used in classifier free guidance...Higher guidance scale encourages the model to generate samples that are more closely linked to the input prompt
        do_sample=True,  # sampling leads to significantly better results than greedy (do_sample=False)
    )
    print('###################', audio_values.shape)

    sampling_rate = model.config.audio_encoder.sampling_rate
    # scipy.io.wavfile.write("0_" + input_text[-1], rate=sampling_rate, data=audio_values[0].cpu())
    # scipy.io.wavfile.write("1_" + input_text[-1], rate=sampling_rate, data=audio_values[1].cpu())  #
    scipy.io.wavfile.write(input_text[-1], rate=sampling_rate, data=audio_values[0, 0].numpy())
