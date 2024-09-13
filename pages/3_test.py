import streamlit as st
import cv2
import time

# 範例
# if "button1" not in st.session_state:
#     st.session_state["button1"] = False

# if "button2" not in st.session_state:
#     st.session_state["button2"] = False

# if "button3" not in st.session_state:
#     st.session_state["button3"] = False

# if st.button("Button1"):
#     st.session_state["button1"] = not st.session_state["button1"]

# if st.session_state["button1"]:
#     if st.button("Button2"):
#         st.session_state["button2"] = not st.session_state["button2"]

# if st.session_state["button1"] and st.session_state["button2"]:
#     if st.button("Button3"):
#         # toggle button3 session state
#         st.session_state["button3"] = not st.session_state["button3"]

# if st.session_state["button3"]:
#     st.write("**Button3!!!**")

# st.write(
#     f"""
#     ## Session state:
#     {st.session_state["button1"]=}

#     {st.session_state["button2"]=}

#     {st.session_state["button3"]=}
#     """
# )

# button_container = st.empty()
# finish = button_container.button("Finish", key  = i)

if "finish" not in st.session_state:
    st.session_state["finish"] = False

if st.button("Search"):
    cap = cv2.VideoCapture(0)
    image_container = st.empty()
    if st.checkbox("Finish"):
        st.session_state["finish"] = not st.session_state["finish"]
    while True:
        ret, frame = cap.read()
        BGR = frame[...,::-1]
        image_container.image(BGR)
        if st.session_state["finish"]:
            break