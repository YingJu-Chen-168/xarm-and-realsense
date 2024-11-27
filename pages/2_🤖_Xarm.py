import streamlit as st
import numpy as np
import cv2
import time
import pyrealsense2 as rs
from ultralytics import YOLO
from scipy.optimize import leastsq
from decimal import Decimal, ROUND_HALF_UP
from xarm_movement.xarm_detection import RobotDetection
from xarm_movement.xarm_search import RobotSearch
from xarm_movement.xarm_treatment import RobotTreatment
from xarm_movement.xarm_gohome import RobotGoHome
from xarm.wrapper import XArmAPI

def check_point(point_x, point_y):
    if point_x >= 0 and point_x < depth_image.shape[1] and point_y >= 0 and point_y < depth_image.shape[0]:
        depth = depth_frame.get_distance(point_x, point_y)
    else:
        depth = None
    return depth
def get_length(position_1, position_2):
    length = np.linalg.norm(np.array(position_1) - np.array(position_2))
    length = int(length*1000)
    return length
def count_area(contours):
    count_area = []
    for i in range(len(contours)):
        count_area.append(cv2.contourArea(contours[i]))
        max_idx = np.argmax(np.array(count_area))
    return max_idx
def sampling_rate(max_contours):
    sampling_contours = []
    for i in range(len(max_contours)):
        if i % 20 == 0:
            xi = max_contours[i][0][0]
            yi = max_contours[i][0][1]
            point = [xi, yi]
            sampling_contours.append(point)
    return sampling_contours
def count_pts(sampling_contours):
    pts = []
    for i in range(len(sampling_contours)):
        xi = sampling_contours[i][0]
        yi = sampling_contours[i][1]
        depth = check_point(xi + 320, yi + 40)
        if depth != None:
            point = rs.rs2_deproject_pixel_to_point(intrinsics, [xi + 320, yi + 40], depth)
            point = [i * 100 for i in point]
            pts.append(point)
    return pts
def round_float(f_num):
    f_num = Decimal(f_num).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
    f_num = float(f_num)
    return f_num
def fit_func(p, x, y):
    a, b, c = p
    return a * x + b * y + c
def residuals(p, x, y, z):
    return z - fit_func(p, x, y)
def estimate_plane_with_leastsq(pts):
    p0 = [1, 0, 1]
    np_pts = np.array(pts)
    plsq = leastsq(residuals, p0, args=(np_pts[:, 0], np_pts[:, 1], np_pts[:, 2]))
    return plsq[0]
def get_proper_plane_params(p, pts):
    assert isinstance(pts, list), r'輸入的數據類型必須依賴 list'
    np_pts = np.array(pts)
    new_pts = []
    for i in range(len(np_pts)):
        new_z = fit_func(p, np_pts[i, 0], np_pts[i, 1])
        new_z = round_float(new_z)
        new_pt = [np_pts[i, 0], np_pts[i, 1], new_z]
        new_pts.append(new_pt)
    if np.linalg.norm(p) < 1e-10:
        print(r'plsq 的 norm 值為 0 {}'.format(p))
    plane_normal = p / np.linalg.norm(p)
    return new_pts, plane_normal
def compute_3D_polygon_area(points):
    if (len(points) < 3): 
        return 0.0
    P1X, P1Y, P1Z = points[0][0], points[0][1], points[0][2]
    P2X, P2Y, P2Z = points[1][0], points[1][1], points[1][2]
    P3X, P3Y, P3Z = points[2][0], points[2][1], points[2][2]
    a = pow(((P2Y-P1Y)*(P3Z-P1Z)-(P3Y-P1Y)*(P2Z-P1Z)), 2) + pow(((P3X-P1X)*(P2Z-P1Z)-(P2X-P1X)*(P3Z-P1Z)), 2) + pow(((P2X-P1X)*(P3Y-P1Y)-(P3X-P1X)*(P2Y-P1Y)), 2)
    cosnx = ((P2Y-P1Y)*(P3Z-P1Z)-(P3Y-P1Y)*(P2Z-P1Z)) / (pow(a, 1/2)) 
    cosny = ((P3X-P1X)*(P2Z-P1Z)-(P2X-P1X)*(P3Z-P1Z)) / (pow(a, 1/2))
    cosnz = ((P2X-P1X)*(P3Y-P1Y)-(P3X-P1X)*(P2Y-P1Y)) / (pow(a, 1/2))
    s = cosnz*((points[-1][0])*(P1Y)-(P1X)*(points[-1][1])) + cosnx*((points[-1][1])*(P1Z)-(P1Y)*(points[-1][2])) + cosny*((points[-1][2])*(P1X)-(P1Z)*(points[-1][0]))
    for i in range(len(points)-1):
        p1 = points[i]
        p2 = points[i+1]
        ss = cosnz*((p1[0])*(p2[1])-(p2[0])*(p1[1])) + cosnx*((p1[1])*(p2[2])-(p2[1])*(p1[2])) + cosny*((p1[2])*(p2[0])-(p2[2])*(p1[0]))
        s += ss 
    s = abs(s/2.0)
    return s
def robot_search():
    arm = XArmAPI('192.168.1.222', baud_checkset=False)
    robot_main = RobotSearch(arm)
    robot_main.run()
def robot_object( X_distance, Y_distance, Z_distance):
    arm = XArmAPI('192.168.1.222', baud_checkset=False)
    robot_main = RobotDetection(arm, Y_distance - 25, X_distance, -(Z_distance - 100))
    robot_main.run()
def robot_treatment(Repeat, Width):
    arm = XArmAPI('192.168.1.222', baud_checkset=False)
    robot_main = RobotTreatment(arm, Repeat, Width)
    robot_main.run()
def robot_gohome():
    arm = XArmAPI('192.168.1.222', baud_checkset=False)
    robot_main = RobotGoHome(arm)
    robot_main.run()
#####################
def all_stop():
    st.session_state.all_stop = True
    robot_gohome()
def all_start():
    robot_search()
    time.sleep(1)
    st.session_state.all_start = True
    st.session_state.capture = False
    st.session_state.detection = False
    st.session_state.treatment = False
    st.session_state.go_home = False
def capture_again():
    robot_search()
    time.sleep(1)
    st.session_state.capture = False
def start_from_the_begining():
    st.session_state.all_start = False
    st.session_state.all_stop = False
    st.session_state.capture = False
    st.session_state.detection = False
    st.session_state.treatment = False
    st.session_state.go_home = False
def submit():
    st.session_state.response = st.session_state.from_widget
    st.session_state.capture = True
def detection():
    st.session_state.detection = True
def treatment():
    st.session_state.treatment = True
def go_home():
    st.session_state.go_home = True

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

W, H = 1280, 720
w, h, x, y, = 640, 640, 320, 40
config = rs.config()
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
pipeline = rs.pipeline()
profile = pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)
model = YOLO('yolov8n-seg.pt')

if st.session_state.all_start:
    if not st.session_state.all_stop:
        slate = st.empty()
        temp = slate.container()
        if not st.session_state.go_home:
            if not st.session_state.treatment:
                if not st.session_state.detection:
                    if not st.session_state.capture:
                        with temp:
                            notice_container = st.empty()
                            distance_container = st.empty()
                            image_container = st.empty()
                            area_container = st.empty()
                            size_container = st.empty()
                            repeat_container = st.empty()
                            notice_container.markdown('The depth camera is capturing.')
                            time.sleep(1)
                            for i in range(50):
                                frames = pipeline.wait_for_frames()
                                aligned_frames = align.process(frames)
                                color_frame = aligned_frames.get_color_frame()
                                depth_frame = aligned_frames.get_depth_frame()
                                color_image = np.asanyarray(color_frame.get_data())
                                color_image = color_image[...,::-1]
                                depth_image = np.asanyarray(depth_frame.get_data())
                                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)
                                crop_color_img = color_image[y:y+h, x:x+w]
                                crop_depth_img = depth_colormap[y:y+h, x:x+w]
                                results = model(crop_color_img, classes = 67)
                                annotated_frame = results[0].plot()
                                intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
                                image_container.image(annotated_frame)
                                if results[0].masks is not None:
                                    mask = (results[0].masks.data[0].to('cpu').detach().numpy().copy() * 255).astype('uint8')
                                    contours,hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                                    max_idx = count_area(contours)
                                    max_contours = contours[max_idx]
                                    sampling_contours = sampling_rate(max_contours)
                                    pts = count_pts(sampling_contours)      
                                    p = estimate_plane_with_leastsq(pts)
                                    polygon, normal = get_proper_plane_params(p, pts)
                                    area = compute_3D_polygon_area(polygon)
                                    
                                    b = results[0].boxes[0].xyxy[0].to('cpu').detach().numpy().copy()  # get box coordinates in (top, left, bottom, right) format
                                    x1 = int(b[0])
                                    y1 = int(b[1])
                                    x2 = int(b[2])
                                    y2 = int(b[3])

                                    depth1 = check_point(x1 + x, y1 + y)
                                    depth2 = check_point(x2 + x, y1 + y)
                                    depth3 = check_point(x2 + x, y2 + y)
                                    if depth1 != None and depth2 != None and depth3 != None:
                                        point1 = rs.rs2_deproject_pixel_to_point(intrinsics, [x1 + x, y1 + y], depth1)
                                        point2 = rs.rs2_deproject_pixel_to_point(intrinsics, [x2 + x, y1 + y], depth2)
                                        point3 = rs.rs2_deproject_pixel_to_point(intrinsics, [x2 + x, y2 + y], depth3)
                                        width = get_length(point2, point1)
                                        height = get_length(point3, point2)
                                        repeat = int(round(height / 30)) + 1
                                        depth_middle = check_point(int(W/2), int(H/2))  # 用畫面的中心點來表示手臂目前的位置
                                        point_middle = rs.rs2_deproject_pixel_to_point(intrinsics, [int(W/2), int(H/2)], depth_middle)
                                        x_distance = int((point_middle[0] - point1[0]) *1000)
                                        y_distance = int((point_middle[1] - point1[1]) *1000)
                                        z_distance = int(depth_middle*1000)
                                        information = [x_distance, y_distance, z_distance,  repeat, width]
                                        information = ' '.join(str(num) for num in information)

                                    area_container.markdown(f'area = {area} cm2')
                                    size_container.markdown(f"width = {width} mm, height = {height} mm")
                                    distance_container.markdown(f'x_distance = {x_distance} mm, y_distance = {y_distance} mm, z_distance = {z_distance} mm')
                                    repeat_container.markdown(f"repeat time = {repeat}")
                                    time.sleep(0.05)
                            notice_container.empty()
                            image_container.empty()
                            st.text_input('(x, y, z) distance + width + repeat', value = information, key = 'from_widget')
                            st.checkbox("Check", on_change = submit)
                    else:
                        with temp:
                            st.button('👉 Go to the Object 👩‍⚕️', on_click = detection)
                            st.button('Capture again', on_click = capture_again)
                            st.button('Break', on_click = all_stop)       
                else:
                    with temp:
                            information = st.session_state.response.split()
                            st.write(f'(x, y, z) distance is ({int(information[0])}, {int(information[1])}, {int(information[2])})')
                            st.button('Start ▶️', on_click = robot_object(int(information[0]), int(information[1]), int(information[2])))
                            
                    with temp:
                            st.button('👉 Go to the Treatment 💉', on_click = treatment)
                            st.button('Capture again', on_click = capture_again)
                            st.button('Break', on_click = all_stop)
            else:
                with temp:
                    information = st.session_state.response.split()
                    st.write(f'Repeat is {int(information[3])}, width is {int(information[4])}')
                    st.button('Start ▶️', on_click = robot_treatment(int(information[3]), int(information[4])))
                with temp:
                    st.button('👉 Go home 👏', on_click = go_home)
                    st.button('Capture again', on_click = capture_again)
                    st.button('Break', on_click = all_stop)
        else:
            with temp:
                st.button('Start ▶️', on_click = robot_gohome())
            with temp:
                st.write("Finish")
                st.button('Capture again', on_click = capture_again)
                st.button('Break', on_click = all_stop)
    else:
        st.write("It's over.")
        st.button('Start from the begining', on_click = start_from_the_begining)
else:
    st.write("Press the button to raise the robot.")
    st.button('👉 Search 🔎', on_click = all_start)