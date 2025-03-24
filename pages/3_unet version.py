from fastapi import FastAPI
from fastapi.responses import StreamingResponse, Response
import numpy as np
import cv2
import pyrealsense2 as rs
from scipy.optimize import leastsq
from decimal import Decimal, ROUND_HALF_UP
from xarm_movement.xarm_search import search_run
from xarm_movement.plasma_gohome import go_home_run
from xarm_movement.xarm_detection import detection_run
from xarm_movement.xarm_treatment import treatment_run
from pydantic import BaseModel
import serial
import time
import math
from model_package.unet import Unet
from model_package.utils import dice_coef, iou_coef
from keras.metrics import Precision, Recall
import tensorflow as tf

app = FastAPI()
information = "0"
wound_image = []

def count_area(contours):
    count_area = []
    for i in range(len(contours)):
        count_area.append(cv2.contourArea(contours[i]))
        max_idx = np.argmax(np.array(count_area))
    return max_idx
def sampling_rate(max_contours):
    sampling_contours = []
    for i in range(len(max_contours)):
        if i % 10 == 0:
            xi = max_contours[i][0][0]
            yi = max_contours[i][0][1]
            point = [xi, yi]
            sampling_contours.append(point)
    return sampling_contours
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
    if len(points) < 3:
        return 0.0
    normal = np.zeros(3)
    q = points[-1]
    for p in points:
        normal += np.cross(q, p)  # 累加外積向量
        q = p
    return np.linalg.norm(normal) / 2.0  # 面積為外積向量長度的一半
def get_coordinate(mask):
    x = []
    y = []
    for i in range(640):
        for j in range(640):
            if mask[i, j] == 1:
                x.append(i)
                y.append(y)
    x1 = min(x)
    y1 = min(y)
    x2 = max(x)
    y2 = max(y)
    return x1, y1, x2, y2
def get_length(position_1, position_2):
    length = np.linalg.norm(np.array(position_1) - np.array(position_2))
    length = int(length*1000)
    return length
def realsense_streaming(mode):
    global information, wound_image
    W, H = 1280, 720
    w, h, x, y, = 640, 640, 320, 40
    M2_diameter = 2
    working_diatance = 153  
    config = rs.config()
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)
    features =  24
    levels =  5
    learning_rate = 9e-4 
    SIZE = 320
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    model = Unet((SIZE, SIZE, 3),  features, levels)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[ dice_coef, iou_coef, Precision(), Recall()])
    model.load_weights("my_model.h5", by_name=True)
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        crop_color_img = color_image[y:y+h, x:x+w]
        show_image = crop_color_img # no_detection的時候
        crop_color_img_rgb = crop_color_img[...,::-1]
        depth_frame = aligned_frames.get_depth_frame()
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        if mode == "detection":
            image = np.float32(np.array(crop_color_img_rgb) / 255)
            image = tf.image.resize(image, (SIZE, SIZE))
            image = tf.expand_dims(image, 0)
            y_pred = model.predict(image)
            if np.all(y_pred == 0): # 要再測試
                y_pred = tf.image.resize(y_pred, [640, 640] )
                y_pred = y_pred.numpy()
                y_pred = 1*(y_pred > 0.5)
                y_pred = np.uint8( y_pred[0]*255 )
                mask = y_pred[:, :, 0]
                contours,hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                max_idx = count_area(contours)
                max_contours = contours[max_idx]
                sampling_contours = sampling_rate(max_contours)
                pts = []
                for i in range(len(sampling_contours)):
                    xi = sampling_contours[i][0]
                    yi = sampling_contours[i][1]
                    # check_point()
                    depth = depth_frame.get_distance(xi + x, yi + y)
                    if depth != 0 and depth != None:
                        point = rs.rs2_deproject_pixel_to_point(intrinsics, [xi + x, yi + y], depth)
                        point = [i * 100 for i in point]
                        pts.append(point)   
                p = estimate_plane_with_leastsq(pts)
                polygon, normal = get_proper_plane_params(p, pts)
                area = compute_3D_polygon_area(polygon)
                zeros = np.zeros((crop_color_img.shape), dtype=np.uint8)
                color_mask = cv2.fillPoly(zeros, contours, color=(0, 165, 255))
                mask_img = cv2.addWeighted(crop_color_img, 0.9, color_mask, 0.3, 0)
                coordinate = get_coordinate(mask)
                x1 = int(coordinate[0])
                y1 = int(coordinate[1])
                x2 = int(coordinate[2])
                y2 = int(coordinate[3])
                cv2.rectangle(mask_img, (x1, y1), (x2, y2), (255, 165, 0), 2)
                # check_point()
                depth1 = depth_frame.get_distance(x1 + x, y1 + y)
                depth2 = depth_frame.get_distance(x2 + x, y1 + y)
                depth3 = depth_frame.get_distance(x2 + x, y2 + y)
                if depth1 != None and depth2 != None and depth3 != None:
                    point1 = rs.rs2_deproject_pixel_to_point(intrinsics, [x1 + x, y1 + y], depth1)
                    point2 = rs.rs2_deproject_pixel_to_point(intrinsics, [x2 + x, y1 + y], depth2)
                    point3 = rs.rs2_deproject_pixel_to_point(intrinsics, [x2 + x, y2 + y], depth3)
                    width = get_length(point2, point1)
                    length = get_length(point3, point2)
                    path_cycle = math.ceil(length / (M2_diameter))   # 看要不要改成 /2，讓路徑比較貼合真實範圍
                    # path_cycle = int(round(length / (2*M2_diameter)))
                    total_working_distance = path_cycle*(width + M2_diameter)
                    plasma_cycle = math.ceil(total_working_distance/working_diatance)
                    # check_point()
                    depth_middle = depth_frame.get_distance(int(W/2), int(H/2)) 
                    point_middle = rs.rs2_deproject_pixel_to_point(intrinsics, [int(W/2), int(H/2)], depth_middle)
                    x_distance = int((point_middle[0] - point1[0]) *1000)
                    y_distance = int((point_middle[1] - point1[1]) *1000)
                    depth_comparison = []
                    for i in range(x1, x2+1):
                        for j in range(y1, y2+1):
                            depth = depth_frame.get_distance(i + x, j + y)
                            if depth != 0 and depth != None:
                                depth_comparison.append(depth)
                    min_idx = np.argmin(np.array(depth_comparison))
                    z_distance = int(depth_comparison[min_idx] *1000)
                    # z_distance = int(depth1 *1000)                 
                    cv2.putText(mask_img, f"(x, y, z) distance is ({x_distance}, {y_distance}, {z_distance}) mm.", 
                                (10, 520), cv2.FONT_HERSHEY_COMPLEX, 0.9, (200, 240, 30), 1)
                    cv2.putText(mask_img, f"Area is {area} cm2.", 
                                (10, 560), cv2.FONT_HERSHEY_COMPLEX, 0.9, (200, 240, 30), 1)
                    cv2.putText(mask_img, f"Width is {width} mm. Length is {length} mm.", 
                                (10, 600), cv2.FONT_HERSHEY_COMPLEX, 0.9, (200, 240, 30), 1)
                    show_image = mask_img   # detection的時候
                    wound_image = show_image
                    # mask_img = results[0].plot()
                    # show_image = mask_img
                    information_list = [x_distance, y_distance, z_distance,  width, path_cycle, plasma_cycle]
                    information = ' '.join(str(num) for num in information_list)
        time.sleep(0.05)
        _, buffer = cv2.imencode('.jpg', show_image)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
def camera_streaming():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
def tcp_run(mode, motor, plasma_cycle):   # 之後要加上激發電漿的程式
    COM_PORT = 'COM3'  
    BAUD_RATES = 9600
    ser = serial.Serial(COM_PORT, BAUD_RATES) 
    if mode == "Reset motor":
        time.sleep(1)
        selected_motor = "X" 
        command = f"Home{selected_motor}\n"  
        ser.write(command.encode('ascii'))
        time.sleep(1)
        selected_motor = "Y" 
        command = f"Home{selected_motor}\n"  
        ser.write(command.encode('ascii'))
    if mode == "Turn plasma":
        if motor == "X":
            time.sleep(1)
            selected_motor = "X" 
            command = f"Turn{selected_motor}\n"
            ser.write(command.encode('ascii'))
        else:
            time.sleep(1)
            selected_motor = "Y"
            command = f"Turn{selected_motor}\n"
            ser.write(command.encode('ascii'))
    if mode == "Light plasma":
        time.sleep(1)
        selected_motor = "X"
        command = f"PlasmaOn{selected_motor} {plasma_cycle}\n"  
        ser.write(command.encode('ascii'))


@app.get("/realsense_streaming")
def streaming(mode: str = "no_detection"):
    return StreamingResponse(realsense_streaming(mode), media_type="multipart/x-mixed-replace; boundary=frame") 
    # multipart/x-mixed-replace：這是一種特殊的 MIME 類型，用於傳輸一連串的資料片段（例如：逐幀圖片）。  

@app.get("/camera_streaming")
def other_streaming():
    return StreamingResponse(camera_streaming(), media_type="multipart/x-mixed-replace; boundary=frame") 

@app.get("/get_information")
def get_information():
    return {"information": information}

@app.get("/get_wound_image")
def get_wound_image():
    _, buffer = cv2.imencode(".jpg", wound_image)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")

@app.get("/xarm_search")
def robot_search():
    search_run()

@app.get("/xarm_gohome")
def robot_gohome():
    go_home_run()

class Detection_Input(BaseModel):
    x_distance : int
    y_distance : int
    z_distance : int
@app.post("/xarm_detection")
def robot_detection(input : Detection_Input):
    detection_run(input.x_distance, input.y_distance, input.z_distance)

class Treatment_Input(BaseModel):
    width : int
    repeat : int
@app.post("/xarm_treatment")
def robot_treatment(input : Treatment_Input):
    treatment_run(input.width, input.repeat)

class Tcp_Input(BaseModel):
    mode : str
    motor : str
    plasma_cycle : int    
@app.post("/arduino")
def tcp(input : Tcp_Input):
    tcp_run(input.mode, input.motor, input.plasma_cycle)

# uvicorn Home_api:app --reload