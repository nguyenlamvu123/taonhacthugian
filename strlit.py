import os
import streamlit as st

from mus_gen import Py_Transformer, Py_Audiocraft, readfile, sampling_rate, historyfile, timer


def main_loop_strl():
    def rendhtmlaudio():
        html: str = ''
        histlist: list = readfile(file=historyfile)
        for ih in range(0, len(histlist), 2):
            out___mp4, b64 = histlist[ih], histlist[ih + 1]
            html += f"""<h5>{out___mp4}</h5>
            <audio controls>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            <a download={out___mp4} href="data:audio/mp3;base64,{b64}">Download</a>
            """
        return html

    def dehi():
        if os.path.isfile(historyfile):
            os.remove(historyfile)

    @timer
    def showdata():
        # st.subheader("This app allows you to find threshold to convert color Image to binary Image!")
        # st.text("We use OpenCV and Streamlit for this demo")

        # horizontal and center radio buttons
        st.write(
            '<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>',
            unsafe_allow_html=True
        )
        st.write(
            '<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>',
            unsafe_allow_html=True
        )

        # Inject custom CSS to set the width of the sidebar
        st.write(
            """
            <style>
                section[data-testid="stSidebar"] {
                    width: 1500px !important; # Set the width to your desired value
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # show at the end of page
        st.write(  # https://stackoverflow.com/questions/41732055/how-to-set-the-div-at-the-end-of-the-page
            """
            <style>
                .banner {
                  width: 100%;
                  height: 15%;
                  position: fixed;
                  bottom: 0;
                  overflow:auto;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

    @timer
    def showdata_col2():
        descri2 = st.text_input("descri2", None)
        thoigian = st.slider(
            "Time to generate music",
            min_value=5,
            max_value=300,
            value=8
        )
        uploaded_file = st.file_uploader("Choose a file")
        return thoigian, descri2, uploaded_file

    @timer
    def showdata_col3():
        descri3 = st.text_input("descri3", None)
        return descri3

    st.title("audiocraft MusicGen")
    showdata()
    col1, col2, col3 = st.columns(3)

    st.sidebar.button('xóa lịch sử', on_click=dehi)
    try:
        md = rendhtmlaudio()
    except AssertionError:
        md = "Xin chào!"
    st.markdown(  # đọc và hiện lịch sử
        f'<div class="banner">{md}</div>',
        unsafe_allow_html=True,
    )

    with col3:
        descri3 = showdata_col3()
        with st.form("checkboxes", clear_on_submit=True):
            # option = st.selectbox(
            #     "How would you like to be contacted?",
            #     ("Py_Transformer", "Py_Audiocraft",),
            #     index=0,
            # )
            g_scale = st.slider(
                "To be used in classifier free guidance (CFG), setting the weighting between the conditional logits (which are predicted from the text prompts) and the unconditional logits (which are predicted from an unconditional or ‘null’ prompt). Higher guidance scale encourages the model to generate samples that are more closely linked to the input prompt, usually at the expense of poorer audio quality. CFG is enabled by setting guidance_scale > 1",
                min_value=0,
                max_value=20,
                value=3
            )
            submit = st.form_submit_button("Run!")  # https://blog.streamlit.io/introducing-submit-button-and-forms/

    with col1:
        descri1 = st.text_input("descri1", None)

    with col2:
        thoigian, descri2, uploaded_file = showdata_col2()

    if not submit:
        return None
    descri = list()
    for s in (descri1, descri2, descri3, ):
        if s is not None: descri += [s]
    pytran = False  # pytran = True if option == "Py_Transformer" else False
    placeholder = st.empty()
    with placeholder.container():
        main_loop(descri, g_scale, thoigian, None, pytran, uploaded_file)


def streamlit_audio(out___mp4_, audio_values, descri):
    data = readfile(file=out___mp4_, mod="rb")
    st.audio(audio_values, sample_rate=sampling_rate)
    for de in descri:
        st.write(de)
    st.download_button(
        label="Download",
        data=data,
        file_name=out___mp4_,
        mime='wav',
    )


def main_loop(descri: list, g_scale, thoigian, outlocat: str or None = None, pytran: bool = True, uploaded_file=None):
    """
    :param descri:
    :param g_scale:
    :param thoigian:
    :param outlocat: '___' when method is called from gradio, or None when method is called from streamlit
    :param pytran:
    :param uploaded_file:
    :return:
    """
    """TODO mark files writen by using datetime instead of using listfilenames (https://www.freecodecamp.org/news/strftime-in-python/) (https://strftime.org/)"""
    if uploaded_file is not None:
        if outlocat is None:  # method is called from streamlit
            cont = uploaded_file.getbuffer()
        else:  # method is called from gradio
            assert outlocat == '___'  # method is called from gradio
            cont = readfile(file=uploaded_file, mod="rb")  # name = Path(uploaded_file).name
        sameaud = f'temp{os.path.splitext(uploaded_file.name)[-1]}'
        readfile(file=sameaud, mod="wb", cont=cont)
    else:
        sameaud = 'temp.wav'
    gener = Py_Transformer(
        input_text=descri, g_scale=int(g_scale), thoigian=int(thoigian), outlocat=outlocat, met="Py_Transformer"
    ) if pytran else Py_Audiocraft(
        input_text=descri, sameaud=sameaud,thoigian=int(thoigian), outlocat=outlocat, met="Py_Audiocraft"
    )
    for out___mp4_, audio_values in gener:
        if outlocat == '___':  # method is called from gradio
            yield out___mp4_, audio_values, descri
        elif outlocat is not None:  # method is called from request
            yield '_'  # TODO
        else:  # method is called from streamlit
            yield streamlit_audio(out___mp4_, audio_values, descri)


if __name__ == '__main__':
    main_loop_strl()  # streamlit run strlit.py --server.port 8501
    # python3 manage.py runserver 0.0.0.0:8501
