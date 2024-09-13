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

st.set_page_config(
    page_title="Xarm",
    page_icon="🤖",
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
        depth = depth_frame.get_distance(point_x, point_y)  # 獲取點的深度值
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
#####################

x_distance, y_distance, z_distance, width, height, repeat = None, None, None, None, None, None

# 機器人抬頭 + 辨識
height = st.number_input("Key in the height you want robot to look up at.", max_value = 550, value = 250, placeholder = "height should between 100 mm and 550 mm.")
if height != None:
    height = int(height)
    st.write("The height is ", height)

if "finish" not in st.session_state:
    st.session_state["finish"] = False

if st.button("Search"):
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
    clss_container = st.empty()
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

        results = model(crop_color_img)
        annotated_frame = results[0].plot()
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0].to('cpu').detach().numpy().copy()  # get box coordinates in (top, left, bottom, right) format
                c = box.cls
                x1 = int(b[0])
                y1 = int(b[1])
                x2 = int(b[2])
                y2 = int(b[3])
                clss = int(c)   # class
                with clss_container.container():
                    st.write("ID = ", clss, "clss = ", model.names[clss])
                if clss == 0:
                    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
                    # 確定座標值是否在depth_image的範圍之内
                    depth1 = check_point(x1+320, y1+40)
                    depth2 = check_point(x2+320, y1+40)
                    depth3 = check_point(x2+320, y2+40)
                    if depth1 != None and depth2 != None and depth3 != None:
                        point1 = rs.rs2_deproject_pixel_to_point(intrinsics, [x1+320, y1+40], depth1)
                        point2 = rs.rs2_deproject_pixel_to_point(intrinsics, [x2+320, y1+40], depth2)
                        point3 = rs.rs2_deproject_pixel_to_point(intrinsics, [x2+320, y2+40], depth3)

                        width = get_length(point2, point1)
                        height = get_length(point3, point2)
                        repeat = int(round(height/30))+1

                        depth4 = check_point(int(W/2), int(H/2))
                        point4 = rs.rs2_deproject_pixel_to_point(intrinsics, [int(W/2), int(H/2)], depth4)
                        x_distance = int((point4[0] - point1[0]) *1000)
                        y_distance = int((point4[1] - point1[1]) *1000)
                        z_distance = int(depth4*1000)
                        with size_container.container():
                            st.write("寬 is:", width, "mm. 長 is:", height, "mm. ", " repeat = ", repeat, 
                                    ", x_distance = ", x_distance, ", y_distance = ", y_distance, ", z_distance = ", z_distance)
                    else:
                        continue
                else:
                    continue
        image_container.image(annotated_frame)
        if st.session_state["finish"]:
            break
else:
    st.caption("Press the button to look up.")
st.divider()
#####################

# 機器人去目標
x_distance = st.number_input("Key in the x_distance", max_value = 550, value = x_distance, placeholder = "x_distance should between 0 mm and 550 mm.")
if x_distance != None:
    x_distance = int(x_distance)
    st.write("The x_distance is ", x_distance)
y_distance = st.number_input("Key in the y_distance", max_value = 300, value = y_distance, placeholder = "y_distance should between 0 mm and 300 mm.")
if y_distance != None:
    y_distance = int(y_distance)
    st.write("The y_distance is ", y_distance)
z_distance = st.number_input("Key in the z_distance", min_value = 100, max_value = 550, value = z_distance, placeholder = "z_distance should between 100 mm and 550 mm.")
if z_distance != None:
    z_distance = int(z_distance)
    st.write("The z_distance is ", z_distance)

if st.button("Go to the object"):
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
repeat = st.number_input("Key in the repeat", min_value = 1, value = repeat, placeholder = "repeat should larger than 0")
if repeat != None:
    repeat = int(repeat)
    st.write("The repeat is ", repeat)
width = st.number_input("Key in the width", max_value = 500, value = width, placeholder = "width should between 0 mm and 500 mm.")
if width != None:
    width = int(width)
    st.write("The width is ", width)

if st.button("Treatment"):
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
if st.button("Go home"):
    robot_move(Home = True)
else:
    st.caption("Press the button to go to the initial position.")
st.divider()
#####################