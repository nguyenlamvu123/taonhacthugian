import scipy
from coordinate_constant import (
    sample_length2num_tokens,
    processor, device, model, g_scale, debug, sampling_rate, sample_length,
)


def configgg(model):
    model.generation_config.guidance_scale = 4.0
    model.generation_config.max_length = 256
    return model


def Py_Transformer_uncondition(num_tokens=256):
    unconditional_inputs = model.get_unconditional_inputs(num_samples=1)
    audio_values = model.generate(**unconditional_inputs, do_sample=True, max_new_tokens=num_tokens)
    return audio_values


def Py_Transformer(input_text, g_scale=3, **kwargs):
    num_tokens = sample_length2num_tokens(kwargs['thoigian'])
    inputs = processor(
        text=input_text,
        padding=True,
        return_tensors="pt",
    ).to(device)
    audio_values = model.generate(
        **inputs,
        max_new_tokens=num_tokens,  # defines the length of the generated music piece
        guidance_scale=g_scale,  # controls the creativity level  # The guidance_scale is used in classifier free guidance...Higher guidance scale encourages the model to generate samples that are more closely linked to the input prompt
        do_sample=True,  # enables stochastic sampling, making the generation process more creative  # sampling leads to significantly better results than greedy (do_sample=False)
    )
    # https://blog.unrealspeech.com/deploying-musicgen-with-custom-inference-endpoints-a-comprehensive-guide/
    return audio_values


if __name__ == '__main__':
    for input_text in (
            (["epic movie theme", "sad jazz", ], "emtsj.wav"),
            # (["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums", ], "80spt_90srs.wav"),
            # (["80s blues track with groovy saxophone"], "80sbt.wav"),
            # (["A serene and peaceful piano piece", ], "sapp.wav", ),
    ):
        audio_values = Py_Transformer(
            input_text=input_text[0], g_scale=g_scale, thoigian=30
        )
        if debug:
            print('###################', audio_values.shape)  # it will be `torch.Size([n, 1, 960000])` with n=len(input_text[0])
            # with `torch.Size([1, 1, 960000])`, `audio_values[1, 0].cpu().numpy()` will cause IndexError

        # scipy.io.wavfile.write("0_" + input_text[-1], rate=sampling_rate, data=audio_values[0, 0].cpu().numpy())
        # scipy.io.wavfile.write("1_" + input_text[-1], rate=sampling_rate, data=audio_values[1, 0].cpu().numpy())
        scipy.io.wavfile.write(input_text[-1], rate=sampling_rate, data=audio_values[1, 0].cpu().numpy())
