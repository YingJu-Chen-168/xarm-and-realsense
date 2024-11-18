import streamlit as st
from streamlit_webrtc import webrtc_streamer

st.set_page_config(
    page_title = "Camera",
    page_icon = "📷",
)

st.title("Live stream")
st.write("Choose your camera device and start.")
st.subheader("", divider='violet')

webrtc_streamer(key="example")