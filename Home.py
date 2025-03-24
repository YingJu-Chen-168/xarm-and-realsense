import streamlit as st
import requests
from concurrent.futures import ThreadPoolExecutor
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
import requests
import time
import json
import asyncio

# Arduino
async def arduino_control(inputs):
    try:
        response = requests.post(url="http://127.0.0.1:8000/arduino", data=json.dumps(inputs), timeout=30)
        response.raise_for_status()
        return "TCP completed successfully."
    except requests.Timeout:
        return "TCP failed: The operation timed out."
    except requests.RequestException as e:
        return f"TCP failed: {e}"
async def tcp_control():
    result = await arduino_control(st.session_state.tcp_data)
    st.session_state.tcp_status = result
###
# 機械手臂動作
async def xarm_search():
    try:
        response = requests.get(url="http://127.0.0.1:8000/xarm_search", timeout=30)
        response.raise_for_status()
        return "Robot search completed successfully."
    except requests.Timeout:
        return "Robot search failed: The operation timed out."
    except requests.RequestException as e:
        return f"Robot search failed: {e}"
async def robot_search():
    result = await xarm_search()
    st.session_state.robot_status = result
#
async def xarm_gohome():
    try:
        response = requests.get(url="http://127.0.0.1:8000/xarm_gohome", timeout=30)
        response.raise_for_status()
        return "Robot go home completed successfully."
    except requests.Timeout:
        return "Robot go home failed: The operation timed out."
    except requests.RequestException as e:
        return f"Robot go home failed: {e}"
async def robot_gohome():
    result = await xarm_gohome()
    st.session_state.robot_status = result
#
async def xarm_detection(inputs):
    try:
        response = requests.post(url="http://127.0.0.1:8000/xarm_detection", data=json.dumps(inputs), timeout=30)
        response.raise_for_status()
        return "Robot detection completed successfully."
    except requests.Timeout:
        return "Robot detection failed: The operation timed out."
    except requests.RequestException as e:
        return f"Robot detection failed: {e}"
async def robot_detection():
    task_xarm = asyncio.create_task(xarm_detection(st.session_state.detection_data))
    result = await task_xarm
    st.session_state.tcp_data = {"mode": "Turn plasma", "motor": "X", "plasma_cycle": 0}
    task_tcp = asyncio.create_task(tcp_control())
    await task_tcp
    st.session_state.robot_status = result
#
async def xarm_treatment(inputs):
    try:
        response = requests.post(url="http://127.0.0.1:8000/xarm_treatment", data=json.dumps(inputs), timeout=3000)
        response.raise_for_status()
        return "Robot treatment completed successfully."
    except requests.Timeout:
        return "Robot treatment failed: The operation timed out."
    except requests.RequestException as e:
        return f"Robot treatment failed: {e}"
async def robot_treatment():  # 同時把兩個任務加入線程，要確認有沒有時間差
    task_tcp = asyncio.create_task(tcp_control())
    task_xarm = asyncio.create_task(xarm_treatment(st.session_state.treatment_data))
    await task_tcp
    result = await task_xarm
    st.session_state.robot_status = result
###
# 串流影像
def realsense_streaming(mode):
    return f"http://127.0.0.1:8000/realsense_streaming?mode={mode}"
def update_realsense():
    if st.session_state.video_stream_url:
        st.markdown(
            f"""
            <div style="text-align:center;">
                <img src="{st.session_state.video_stream_url}" alt="Video Stream">
            </div>
            """,
            unsafe_allow_html=True,
        )
def handle_streaming(mode):
    ctx = get_script_run_ctx()
    future_realsense = executor.submit(realsense_streaming, mode)
    add_script_run_ctx(future_realsense, ctx)
    st.session_state.video_stream_url = future_realsense.result()
    update_realsense()
def camera_streaming():
    return f"http://127.0.0.1:8000/camera_streaming"
def update_camera():
    if st.session_state.video_stream_url:
        st.markdown(
            f"""
            <div style="text-align:center;">
                <img src="{st.session_state.video_stream_url}" alt="Video Stream">
            </div>
            """,
            unsafe_allow_html=True,
        )
def handle_other_streaming():
    ctx = get_script_run_ctx()
    future_realsense = executor.submit(camera_streaming)
    add_script_run_ctx(future_realsense, ctx)
    st.session_state.video_stream_url = future_realsense.result()
    update_camera()
###
def fetch_information():
    response = requests.get("http://127.0.0.1:8000/get_information")
    if response.status_code == 200:
        st.session_state.information = response.json().get("information", "N/A")    # N/A = Not Avalible
def fetch_wound_image():
    st.markdown(
        f"""
        <div style="text-align:center;">
            <img src="http://127.0.0.1:8000/get_wound_image" alt="Image" width="500">
        </div>
        """,
        unsafe_allow_html=True,
    )
###
async def all_start():
    st.session_state.tcp_data = {"mode": "Reset motor", "motor": "None", "plasma_cycle": 0}
    task_xarm = asyncio.create_task(robot_search())
    await task_xarm
    task_tcp = asyncio.create_task(tcp_control())
    await task_tcp
def all_start_state():
    st.session_state.all_start = True
    st.session_state.capture = False
    st.session_state.detection = False
    st.session_state.treatment = False
    st.session_state.go_home = False
def submit():
    fetch_information()
    st.session_state.response = st.session_state.information
    st.session_state.capture = True
def detection_state():
    st.session_state.detection = True
def capture_again():
    st.session_state.capture = False
    st.session_state.detection = False
    st.session_state.treatment = False
def all_stop():
    st.session_state.all_stop = True
def treatment_state():
    st.session_state.treatment = True
def go_home_state():
    st.session_state.go_home = True
def start_from_the_beginning():
    st.session_state.all_start = False
    st.session_state.all_stop = False
    st.session_state.capture = False
    st.session_state.detection = False
    st.session_state.treatment = False
    st.session_state.go_home = False
###
# 初始化
if "video_stream_url" not in st.session_state:
    st.session_state.video_stream_url = None
if "robot_status" not in st.session_state:
    st.session_state.robot_status = "Idle"
if "tcp_status" not in st.session_state:
    st.session_state.tcp_status = "Idle"
if "status_message" not in st.session_state:
    st.session_state.status_message = ""
if "information" not in st.session_state:
    st.session_state.information = "0"
if "detection_data" not in st.session_state:
    st.session_state.detection_data = {0}
if "treatment_data" not in st.session_state:
    st.session_state.treatment_data = {0}
if "tcp_data" not in st.session_state:
    st.session_state.tcp_data = {0}
if 'all_stop' not in st.session_state:
    st.session_state.all_stop = False
if 'all_start' not in st.session_state:
    st.session_state.all_start = False
if 'capture' not in st.session_state:
    st.session_state.capture = False
if 'detection' not in st.session_state:
    st.session_state.detection = False
if 'treatment' not in st.session_state:
    st.session_state.treatment = False
if 'go_home' not in st.session_state:
    st.session_state.go_home = False

st.title("👩‍⚕️ GPSR_M5 user interface 🤖")
executor = ThreadPoolExecutor(max_workers=2)
if st.session_state.all_start:
    if not st.session_state.all_stop:
        if not st.session_state.go_home:
            if not st.session_state.treatment:
                if not st.session_state.detection:
                    if not st.session_state.capture:                                                
                        st.subheader("Image Capture")
                        st.write("Connect to realsense...")                        
                        handle_streaming("detection")
                        st.checkbox("Finish capture", on_change = submit)
                    else:
                        st.subheader("Treatment Parameters")
                        fetch_wound_image()
                        information = st.session_state.response.split()
                        path_cycle = int(information[4]) // 2
                        st.write(f'Treatment path cycle is {path_cycle}.')
                        st.write(f'Plasma cycle is {information[5]}.')
                        st.button('👉 Next step 🛖', on_click = detection_state)
                        st.button('Capture again 📷', on_click = capture_again)
                        st.button('Terminate treatment ⏸️', on_click = all_stop)
                else:
                    st.subheader("Positioning")
                    information = st.session_state.response.split()
                    st.write(f'(x, y, z) distance is ({information[0]}, {information[1]}, {information[2]})')
                    st.session_state.detection_data = {"x_distance": int(information[0]), "y_distance": int(information[1]), "z_distance": int(information[2])}
                    st.write("Connect to camera...")  
                    handle_other_streaming()
                    # handle_streaming("no_detection")
                    if st.button('Start ▶️'):
                        asyncio.run(robot_detection())
                    if st.session_state.robot_status == "Robot detection completed successfully.":
                        st.success(f"Robot Status: {st.session_state.robot_status}")
                    else:
                        st.error("Robot Status: Haven't pressed the Start ▶️ button.")
                    st.button('👉 Next step 🚐', on_click = treatment_state)
                    st.button('Terminate treatment ⏸️', on_click = all_stop)                        
            else:
                st.subheader("Plasma Treatment")
                information = st.session_state.response.split()
                path_cycle = int(information[4]) // 2
                st.write(f'Treatment path cycle is {path_cycle}')
                st.write(f'Plasma cycle is {information[5]}.')
                st.session_state.treatment_data = {"width": int(information[3]), "repeat": int(information[4])}
                st.session_state.tcp_data = {"mode": "Light plasma", "motor": "X", "plasma_cycle": int(information[5])}
                st.write("Connect to camera...")  
                handle_other_streaming()
                # handle_streaming("no_detection")
                if st.button('Start ▶️'):
                    asyncio.run(robot_treatment()) 
                if st.session_state.robot_status == "Robot treatment completed successfully.":
                    st.success(f"Robot Status: {st.session_state.robot_status}")
                else:
                    st.error("Robot Status: Haven't pressed the Start ▶️ button.")
                st.button('👉 Next step 👏', on_click = go_home_state)
                st.button('Terminate treatment ⏸️', on_click = all_stop)   
        else:
            st.subheader("Homing")
            st.write("Connect to camera...")  
            handle_other_streaming()
            # handle_streaming("no_detection")
            if st.button('Start ▶️'):
                asyncio.run(robot_gohome()) 
            if st.session_state.robot_status == "Robot go home completed successfully.":
                st.success(f"Robot Status: {st.session_state.robot_status}")
            else:
                st.error("Robot Status: Haven't pressed the Start ▶️ button.")
            st.button('Terminate treatment ⏸️', on_click = all_stop)                
    else:
        st.subheader("Finish ! ")
        if st.session_state.robot_status != "Robot go home completed successfully.":
            asyncio.run(robot_gohome())
        else:
            st.success("Plasma treatment is terminated.")
        st.button('👉 Start another treatment  ❕', on_click = start_from_the_beginning)
else:
    st.subheader("Start !")
    st.write("Connect to camera...")  
    handle_other_streaming()
    st.write("Press the button to start plasma treatment.")
    if st.button('Start ▶️'):
        asyncio.run(all_start())
    if st.session_state.robot_status == "Robot search completed successfully.":
        st.success(f"Robot Status: {st.session_state.robot_status}")
    else:
        st.error("Robot Status: Haven't pressed the Start ▶️ button.")
    st.button('👉 Next step 🔎', on_click = all_start_state)

# streamlit run Home.py
#   .\ssl-proxy.exe -from 0.0.0.0:443 -to localhost:8501
#   https://192.168.50.159