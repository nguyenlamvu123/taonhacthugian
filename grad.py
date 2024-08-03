import gradio as st
from strlit import main_loop, sampling_rate


def main_loop_strl(descri1, descri2, descri3, g_scale, thoigian, uploaded_file):
    global outmp4list
    descri = list()
    for s in (descri1, descri2, descri3, ):
        if s is not None: descri += [s]
    for out___mp4_, audio_values, descri in main_loop(descri, g_scale, thoigian, "___", False, uploaded_file):
        outmp4 = st.Audio(value=(sampling_rate, audio_values, ), label=descri, visible=True)
        outmp4list.append(outmp4)
    return outmp4list

def showdata_col1():
    descri1 = st.Textbox(label="descri1")
    return descri1

def showdata_col2():
    descri2 = st.Textbox(label="descri2")
    thoigian = st.Slider(
        5, 300,
        step=1.0,
        label="Time to generate music",
        value=8,
    )
    uploaded_file = st.UploadButton("Choose a file", file_types=["audio"])
    return thoigian, descri2, uploaded_file

def showdata_col3():
    descri3 = st.Textbox(label="descri3")
    return descri3


outmp4list = list()
if __name__ == '__main__':
    # pytran = False  # pytran = True if option == "Py_Transformer" else False
    with st.Blocks() as demo:
        with st.Row():
            with st.Column():
                descri1 = showdata_col1()
            with st.Column():
                thoigian, descri2, uploaded_file = showdata_col2()
            with st.Column():
                descri3 = showdata_col3()
                g_scale = st.Slider(
                    0, 20,
                    step=1.0,
                    label="To be used in classifier free guidance (CFG), setting the weighting between the conditional logits (which are predicted from the text prompts) and the unconditional logits (which are predicted from an unconditional or ‘null’ prompt). Higher guidance scale encourages the model to generate samples that are more closely linked to the input prompt, usually at the expense of poorer audio quality. CFG is enabled by setting guidance_scale > 1",
                    value=3,
                )
        with st.Row():  # site: https://www.gradio.app/docs/gradio/downloadbutton
            outmp4_1 = st.Audio(visible=False)
            outmp4_2 = st.Audio(visible=False)
            outmp4_3 = st.Audio(visible=False)
        greet_btn = st.Button("Run!")
        greet_btn.click(
            fn=main_loop_strl,
            inputs=[
                descri1,
                descri2,
                descri3,
                g_scale,
                thoigian,
                # pytran,
                uploaded_file,
            ],
            outputs=[outmp4_1, outmp4_2, outmp4_3, ],
            api_name="greet"
        )
    demo.launch()
