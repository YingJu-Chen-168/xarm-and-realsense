import streamlit as st
import cv2
import numpy as np
import time
import pyrealsense2 as rs
from ultralytics import YOLO
from xarm_movement.xarm_detection import RobotDetection
from xarm_movement.xarm_search import RobotSearch
from xarm_movement.xarm_treatment import RobotTreatment
from xarm_movement.xarm_gohome import RobotGoHome
from xarm.wrapper import XArmAPI
from scipy.optimize import leastsq
from decimal import Decimal, ROUND_HALF_UP

st.set_page_config(
    page_title = "Xarm",
    page_icon = "🤖",
)

st.title("Robot xarm")
st.write("Select a xarm movement below.")
st.subheader("", divider='blue')
st.subheader("Recommended steps : Search → Go to the object → Treatment → Go home")
st.divider()

#####################
def reset():
    x_distance = None
    y_distance = None
    z_distance = None
    repeat = None
    width = None
    height = None
    return x_distance, y_distance, z_distance, repeat, width, height
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
def robot_move(Search = False, Object = False, Treatment = False, Home = False, 
               Height = None, X_distance = None, Y_distance = None, Z_distance = None, Repeat = None, Width = None):
    arm = XArmAPI('192.168.1.222', baud_checkset=False)
    if Search:
        robot_main = RobotSearch(arm, Height)
        robot_main.run()
    if Object:
        robot_main = RobotDetection(arm, Y_distance - 25, X_distance, -(Z_distance - 100))
        robot_main.run()
    if Treatment:
        robot_main = RobotTreatment(arm, Repeat, Width)
        robot_main.run()
    if Home:
        robot_main = RobotGoHome(arm)
        robot_main.run()
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
#####################

# 機器人抬頭 + 辨識
height = st.number_input("Please enter the height you want the robot arm to raise.", max_value = 550, value = 400, placeholder = "Height should between 100 mm and 550 mm.")
if height != None:
    height = int(height)
    st.write("The height is ", height)

if "finish" not in st.session_state:
    st.session_state["finish"] = False

if st.button("👉 Search 🔎"):
    robot_move(Search = True, Height = height)
    x_distance, y_distance, z_distance, repeat, width, height = reset()

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
    image_container = st.empty()
    area_container = st.empty()
    size_container = st.empty()
    id_container = st.empty()
    if st.checkbox("Finish video"):
        st.session_state["finish"] = not st.session_state["finish"]

    time.sleep(1)
    while True:
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

        results = model(crop_color_img, classes=0)
        annotated_frame = results[0].plot()
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        if results[0].masks is not None:
            mask = (results[0].masks.data[0].to('cpu').detach().numpy().copy() * 255).astype('uint8')
            contours,hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            count_area = []
            for i in range(len(contours)):
                count_area.append(cv2.contourArea(contours[i]))
            max_idx = np.argmax(np.array(count_area))
            max_contours = contours[max_idx]

            pts = []
            for i in range(len(max_contours)):
                xi = max_contours[i][0][0]
                yi = max_contours[i][0][1]
                depth = check_point(xi + x, yi + y)
                if depth != None:
                    point = rs.rs2_deproject_pixel_to_point(intrinsics, [xi + x, yi + y], depth)
                    point = [i * 100 for i in point]
                    pts.append(point)
                    
            p = estimate_plane_with_leastsq(pts)
            polygon, normal = get_proper_plane_params(p, pts)
            area = compute_3D_polygon_area(polygon)
            with area_container.container():
                st.write("Area of the object = ", area, " cm2.")

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
                with size_container.container():
                    st.write("width = ", width, "mm. height = ", height, "mm. ", " repeat time = ", repeat, 
                            ", x_distance = ", x_distance, ", y_distance = ", y_distance, ", z_distance = ", z_distance)
            else:
                continue
        else:
            continue

        if results[0].masks is None:
            image_container.image(crop_color_img)
        else:
            image_container.image(annotated_frame)

        if st.session_state["finish"]:
            break
else:
    st.caption("Press the button to raise.")
st.divider()
#####################

# 機器人去目標
x_distance = st.number_input("Please enter the x_distance.", max_value = 550, value = None, placeholder = "x_distance should between 0 mm and 550 mm.")
if x_distance != None:
    x_distance = int(x_distance)
    st.write("The x_distance is ", x_distance)
y_distance = st.number_input("Please enter the y_distance.", max_value = 300, value = None, placeholder = "y_distance should between 0 mm and 300 mm.")
if y_distance != None:
    y_distance = int(y_distance)
    st.write("The y_distance is ", y_distance)
z_distance = st.number_input("Please enter the z_distance.", min_value = 100, max_value = 550, value = None, placeholder = "z_distance should between 100 mm and 550 mm.")
if z_distance != None:
    z_distance = int(z_distance)
    st.write("The z_distance is ", z_distance)

if st.button("👉 Go to the object 👩‍⚕️"):
    if x_distance != None and y_distance != None and z_distance >= 100:
        robot_move(Object = True, X_distance = x_distance, Y_distance = y_distance, Z_distance = z_distance)
        x_distance, y_distance, z_distance, repeat, width, height = reset()
    else:
        st.write("At least one distance is not acceptable.")
else:
    st.caption("Press the button to go to the object.")
st.divider()
#####################

# 機器人治療
repeat = st.number_input("Please enter the repeat time.", min_value = 1, value = None, placeholder = "repeat should larger than 0")
if repeat != None:
    repeat = int(repeat)
    st.write("The repeat is ", repeat)
width = st.number_input("Please enter the width.", max_value = 500, value = None, placeholder = "width should between 0 mm and 500 mm.")
if width != None:
    width = int(width)
    st.write("The width is ", width)

if st.button("👉 Treatment 💉"):
    if repeat != None and width != None:
        robot_move(Treatment = True, Repeat = repeat, Width = width)
        x_distance, y_distance, z_distance, repeat, width, height = reset()
    else:
        st.write("At least one value is not acceptable.")
else:
    st.caption("Press the button to do the treatment.")
st.divider()
#####################

# 機器人回家
if st.button("👉 Go home 👏"):
    robot_move(Home = True)
else:
    st.caption("Press the button to go to the initial position.")
st.divider()
#####################