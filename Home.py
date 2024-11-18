import streamlit as st

st.set_page_config(
    page_title="Home page",
    page_icon="🔆",
)

st.title("GPSR_M5 User Interface")
st.write("☆*: .｡. o(≧▽≦)o .｡.:*☆")
st.subheader("", divider='rainbow')

st.markdown(
    """
    ### Student
    - [Ying Ju Chen](https://www.facebook.com/yingjuju.528?locale=zh_TW)
"""
)

st.sidebar.success("Select a demo above.")

#   streamlit run Home.py
#   .\ssl-proxy.exe -from 0.0.0.0:443 -to localhost:8501
#   https://192.168.50.159