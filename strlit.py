import base64, os, scipy
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
        return thoigian, descri2

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

    with col1:
        descri1 = st.text_input("descri1", None)
        with st.form("checkboxes", clear_on_submit=True):
            option = st.selectbox(
                "How would you like to be contacted?",
                ("Py_Transformer", "Py_Audiocraft",),
                index=0,
            )
            g_scale = st.slider(
                "To be used in classifier free guidance (CFG), setting the weighting between the conditional logits (which are predicted from the text prompts) and the unconditional logits (which are predicted from an unconditional or ‘null’ prompt). Higher guidance scale encourages the model to generate samples that are more closely linked to the input prompt, usually at the expense of poorer audio quality. CFG is enabled by setting guidance_scale > 1",
                min_value=0,
                max_value=20,
                value=3
            )
            submit = st.form_submit_button('Chạy!')  # https://blog.streamlit.io/introducing-submit-button-and-forms/

    with col2:
        thoigian, descri2 = showdata_col2()

    if not submit:
        return None
    descri = list()
    for s in (descri1, descri2, descri3, ):
        if s is not None: descri += [s]
    pytran = True if option == "Py_Transformer" else "Py_Audiocraft"
    placeholder = st.empty()
    with placeholder.container():
        main_loop(descri, g_scale, thoigian, None, pytran)


def main_loop(descri: list, g_scale, thoigian, outlocat: str or None = None, pytran: bool = True):
    """TODO mark files writen by using datetime instead of using listfilenames (https://www.freecodecamp.org/news/strftime-in-python/) (https://strftime.org/)"""
    if pytran:
        gener = Py_Transformer(input_text=descri, g_scale=int(g_scale), thoigian=int(thoigian), outlocat=outlocat)
    else:
        Py_Audiocraft(descri, thoigian=int(thoigian), outlocat=outlocat)  # -> 0.wav, 1.wav, 2.wav
    if outlocat is not None:
        return
    for out___mp4_, audio_values in gener:
        data = readfile(file=out___mp4_, mod="rb")
        st.audio(audio_values, sample_rate=sampling_rate)
        b64 = base64.b64encode(data).decode()
        readfile(file=historyfile, mod="a", cont=f'previous time\n{b64}\n')  # ghi lại lịch sử dưới dạng base64 vào file trên local
        for de in descri:
            st.write(de)
        st.download_button(
            label="Download",
            data=data,
            file_name=out___mp4_,
            mime='wav',
        )


if __name__ == '__main__':
    main_loop_strl()  # streamlit run strlit.py --server.port 8501
    # python3 manage.py runserver 0.0.0.0:8501
