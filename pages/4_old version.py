import streamlit as st
import requests
from concurrent.futures import ThreadPoolExecutor
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
import requests
import time
import json

# Arduino
def arduino_control(inputs):
    try:
        response = requests.post(url="http://127.0.0.1:8000/arduino", data=json.dumps(inputs), timeout=30)
        response.raise_for_status()
        return "TCP completed successfully."
    except requests.Timeout:
        return "TCP failed: The operation timed out."
    except requests.RequestException as e:
        return f"TCP failed: {e}"
def tcp_control():
    ctx = get_script_run_ctx()
    future_tcp = executor.submit(arduino_control, st.session_state.tcp_data)
    add_script_run_ctx(future_tcp, ctx)
    st.session_state.tcp_status = "TCP is moving..."
    result = future_tcp.result()
    st.session_state.tcp_status = result
###
# 機械手臂動作
def xarm_search():
    try:
        response = requests.get(url="http://127.0.0.1:8000/xarm_search", timeout=30)
        response.raise_for_status()
        return "Robot search completed successfully."
    except requests.Timeout:
        return "Robot search failed: The operation timed out."
    except requests.RequestException as e:
        return f"Robot search failed: {e}"
def robot_search():
    ctx = get_script_run_ctx()
    future_arm = executor.submit(xarm_search)
    add_script_run_ctx(future_arm, ctx)
    st.session_state.robot_status = "Running..."
    result = future_arm.result()
    st.session_state.robot_status = result
#
def xarm_gohome():
    try:
        response = requests.get(url="http://127.0.0.1:8000/xarm_gohome", timeout=30)
        response.raise_for_status()
        return "Robot go home completed successfully."
    except requests.Timeout:
        return "Robot go home failed: The operation timed out."
    except requests.RequestException as e:
        return f"Robot go home failed: {e}"
def robot_gohome():
    ctx = get_script_run_ctx()
    future_arm = executor.submit(xarm_gohome)
    add_script_run_ctx(future_arm, ctx)
    st.session_state.robot_status = "Running..."
    result = future_arm.result()
    st.session_state.robot_status = result
    st.session_state.movement = True
#
def xarm_detection(inputs):
    try:
        response = requests.post(url="http://127.0.0.1:8000/xarm_detection", data=json.dumps(inputs), timeout=30)
        response.raise_for_status()
        return "Robot detection completed successfully."
    except requests.Timeout:
        return "Robot detection failed: The operation timed out."
    except requests.RequestException as e:
        return f"Robot detection failed: {e}"
def robot_detection():
    ctx = get_script_run_ctx()
    future_arm = executor.submit(xarm_detection, st.session_state.detection_data)
    add_script_run_ctx(future_arm, ctx)
    st.session_state.robot_status = "Running..."
    result = future_arm.result()
    st.session_state.robot_status = result
    st.session_state.movement = True
#
def xarm_treatment(inputs):
    try:
        response = requests.post(url="http://127.0.0.1:8000/xarm_treatment", data=json.dumps(inputs), timeout=3000)
        response.raise_for_status()
        return "Robot treatment completed successfully."
    except requests.Timeout:
        return "Robot treatment failed: The operation timed out."
    except requests.RequestException as e:
        return f"Robot treatment failed: {e}"
def robot_treatment():  # 同時把兩個任務加入線程，要確認有沒有時間差
    # ctx = get_script_run_ctx()
    # future_tcp = executor.submit(arduino_control, st.session_state.tcp_data)
    # add_script_run_ctx(future_tcp, ctx)
    ctx = get_script_run_ctx()
    future_arm = executor.submit(xarm_treatment, st.session_state.treatment_data)
    add_script_run_ctx(future_arm, ctx)
    st.session_state.robot_status = "Running..."
    result = future_arm.result()
    st.session_state.robot_status = result
    st.session_state.movement = True
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
# 動態更新數字
def fetch_information():
    response = requests.get("http://127.0.0.1:8000/get_information")
    if response.status_code == 200:
        st.session_state.information = response.json().get("information", "N/A")    # N/A = Not Avalible
###
def all_start():
    robot_search()
    time.sleep(1)
    st.session_state.tcp_data = {"mode": "Reset motor", "motor": "None", "plasma_cycle": 0}
    tcp_control()
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
    robot_search()
    time.sleep(1)
    st.session_state.capture = False
    st.session_state.detection = False
    st.session_state.treatment = False
    st.session_state.movement = False
def all_stop():
    robot_gohome()
    st.session_state.all_stop = True
def finish_tcp_control():
    st.session_state.movement = True
def treatment_state():
    st.session_state.treatment = True
    st.session_state.movement = False
def go_home_state():
    st.session_state.go_home = True
    st.session_state.movement = False
def start_from_the_beginning():
    st.session_state.all_start = False
    st.session_state.all_stop = False
    st.session_state.capture = False
    st.session_state.detection = False
    st.session_state.treatment = False
    st.session_state.go_home = False
    st.session_state.movement = False
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
if "movement" not in st.session_state:
    st.session_state.movement = False
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
executor = ThreadPoolExecutor(max_workers=4)
if st.session_state.all_start:
    if not st.session_state.all_stop:
        if not st.session_state.go_home:
            if not st.session_state.treatment:
                if not st.session_state.detection:
                    if not st.session_state.capture:                        
                        st.success(f"Robot Status: {st.session_state.robot_status}")                        
                        st.subheader("Image Capture")
                        st.write("Connect to realsense...")                        
                        handle_streaming("detection")
                        st.checkbox("Finish capture", on_change = submit)
                    else:
                        st.subheader("Treatment Parameters")
                        information = st.session_state.response.split()
                        st.write(f'(x, y, z) distance is ({information[0]}, {information[1]}, {information[2]}) mm.')
                        st.write(f'Width is {information[3]} mm.')
                        st.write(f'Treatment path cycle is {information[4]}.')
                        st.write(f'Plasma cycle is {information[5]}.')
                        st.button('👉 Next step 🛖', on_click = detection_state)
                        st.button('Capture again 📷', on_click = capture_again)
                        st.button('Break ⏸️', on_click = all_stop)       
                else:
                    if not st.session_state.movement:
                        st.subheader("Positioning")
                        st.write("Connect to realsense...")
                        handle_streaming("no_detection")
                        information = st.session_state.response.split()
                        st.write(f'(x, y, z) distance is ({information[0]}, {information[1]}, {information[2]})')
                        st.session_state.detection_data = {"x_distance": int(information[0]), "y_distance": int(information[1]), "z_distance": int(information[2])}
                        st.button('Start ▶️', on_click = robot_detection)
                    else:
                        st.subheader("Positioning")
                        st.success(f"Robot Status: {st.session_state.robot_status}")
                        st.button('👉 Next step 🚐', on_click = treatment_state)
                        st.button('Capture again 📷', on_click = capture_again)
                        st.button('Break ⏸️', on_click = all_stop)
            else:
                if not st.session_state.movement:
                    st.subheader("Plasma Treatment")
                    st.write("Connect to realsense...")
                    handle_streaming("no_detection")
                    information = st.session_state.response.split()
                    st.write(f'Width is {information[3]} mm.')
                    st.write(f'Treatment path cycle is {information[4]}')
                    st.write(f'Plasma cycle is {information[5]}.')
                    st.session_state.treatment_data = {"width": int(information[3]), "repeat": int(information[4])}
                    st.session_state.tcp_data = {"mode": "Turn plasma", "motor": "X", "plasma_cycle": 0}
                    tcp_control()
                    # st.session_state.tcp_data = {"mode": "Light plasma", "motor": "X", "plasma_cycle": int(information[6])}
                    st.button('Start ▶️', on_click = robot_treatment)    
                else:
                    st.subheader("Plasma Treatment")
                    st.write("Connect to realsense...")
                    handle_streaming("no_detection")
                    st.success(f"Robot Status: {st.session_state.robot_status}")
                    st.button('👉 Next step 👏', on_click = go_home_state)
                    st.button('Capture again 📷', on_click = capture_again)
                    st.button('Break ⏸️', on_click = all_stop)
        else:
            if not st.session_state.movement:
                st.subheader("Reset")
                st.write("Connect to realsense...")
                handle_streaming("no_detection")
                st.button('Start ▶️', on_click = robot_gohome)
            else:
                st.subheader("Reset")
                st.success(f"Robot Status: {st.session_state.robot_status}")
                st.button('Break ⏸️', on_click = all_stop)
    else:
        st.subheader("Finish ! ")
        st.success("Plasma treatment completed.")
        st.button('👉 Start from the beginning ❕', on_click = start_from_the_beginning)
else:
    st.subheader("Start !")
    st.write("Press the button to raise the robot.")
    st.button('👉 Search 🔎', on_click = all_start)