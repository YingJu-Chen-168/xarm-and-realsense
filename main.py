import cv2
import numpy as np
import time
import pyrealsense2 as rs
from ultralytics import YOLO
from xarm_movement.xarm_detection import RobotDetection
from xarm_movement.xarm_search import RobotSearch
from xarm_movement.xarm_treatment import RobotTreatment
from xarm_movement.xarm_gohome import RobotGoHome
from xarm import version
from xarm.wrapper import XArmAPI
from scipy.optimize import leastsq
from decimal import Decimal, ROUND_HALF_UP

def reset():
    x_distance = None
    y_distance = None
    z_distance = None
    repeat = None
    width = None
    return x_distance, y_distance, z_distance, repeat, width

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
    plsq = leastsq(residuals, p0, args=(np_pts[:, 0], np_pts[:, 1], np_pts[:, 2]))  # p0表示初始的估計參數
    return plsq[0]

def get_proper_plane_params(p, pts):
    assert isinstance(pts, list), r'輸入的數據類型必須依賴 list'    # assert : 斷言, 失敗時出現的訊息
    np_pts = np.array(pts)

    new_pts = []
    for i in range(len(np_pts)):
        new_z = fit_func(p, np_pts[i, 0], np_pts[i, 1])
        new_z = round_float(new_z)
        new_pt = [np_pts[i, 0], np_pts[i, 1], new_z]
        new_pts.append(new_pt)

    if np.linalg.norm(p) < 1e-10:
        print(r'plsq 的 norm 值為 0 {}'.format(p))
    plane_normal = p / np.linalg.norm(p)    # np.linalg.norm(p) 為平方和開根號，這邊應該是在對法向量normaliza

    return new_pts, plane_normal

def compute_3D_polygon_area(points):
    if (len(points) < 3): 
        return 0.0
    
    P1X, P1Y, P1Z = points[0][0], points[0][1], points[0][2]
    P2X, P2Y, P2Z = points[1][0], points[1][1], points[1][2]
    P3X, P3Y, P3Z = points[2][0], points[2][1], points[2][2]

    a = pow(((P2Y-P1Y)*(P3Z-P1Z)-(P3Y-P1Y)*(P2Z-P1Z)), 2) + pow(((P3X-P1X)*(P2Z-P1Z)-(P2X-P1X)*(P3Z-P1Z)), 2) + pow(((P2X-P1X)*(P3Y-P1Y)-(P3X-P1X)*(P2Y-P1Y)), 2)
    # ((P2Y-P1Y)*(P3Z-P1Z)-(P3Y-P1Y)*(P2Z-P1Z)) 的 2次方 + ((P3X-P1X)*(P2Z-P1Z)-(P2X-P1X)*(P3Z-P1Z)) 的 2次方 + ((P2X-P1X)*(P3Y-P1Y)-(P3X-P1X)*(P2Y-P1Y)) 的 2次方
    
    cosnx = ((P2Y-P1Y)*(P3Z-P1Z)-(P3Y-P1Y)*(P2Z-P1Z)) / (pow(a, 1/2))  # a 的 1/2次方
    cosny = ((P3X-P1X)*(P2Z-P1Z)-(P2X-P1X)*(P3Z-P1Z)) / (pow(a, 1/2))
    cosnz = ((P2X-P1X)*(P3Y-P1Y)-(P3X-P1X)*(P2Y-P1Y)) / (pow(a, 1/2))

    s = cosnz*((points[-1][0])*(P1Y)-(P1X)*(points[-1][1])) + cosnx*((points[-1][1])*(P1Z)-(P1Y)*(points[-1][2])) + cosny*((points[-1][2])*(P1X)-(P1Z)*(points[-1][0]))
    # points[-1][0] 的-1指的是最後一個元素

    for i in range(len(points)-1):
        p1 = points[i]
        p2 = points[i+1]
        ss = cosnz*((p1[0])*(p2[1])-(p2[0])*(p1[1])) + cosnx*((p1[1])*(p2[2])-(p2[1])*(p1[2])) + cosny*((p1[2])*(p2[0])-(p2[2])*(p1[0]))
        s += ss 

    s = abs(s/2.0)

    return s

W, H = 1280, 720
w, h, x, y, = 640, 640, 320, 40
x_distance, y_distance, z_distance, repeat, width = reset()
s = 0

config = rs.config()
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
pipeline = rs.pipeline()
profile = pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)

model_directory = 'yolov8n-seg.pt'      #os.environ['HOME'] + '/yolov8_rs/yolov8m.pt'
model = YOLO(model_directory)

robot_move(Search = True)

while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    color_image = np.asanyarray(color_frame.get_data())
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
        print("Area = ", area)

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
            print("width = ", width, "mm. height = ", height, "mm.", "repeat time = ", repeat)

            depth_middle = check_point(int(W/2), int(H/2))  # 用畫面的中心點來表示realsense目前的位置
            point_middle = rs.rs2_deproject_pixel_to_point(intrinsics, [int(W/2), int(H/2)], depth_middle)
            x_distance = int((point_middle[0] - point1[0]) *1000)
            y_distance = int((point_middle[1] - point1[1]) *1000)
            z_distance = int(depth_middle*1000)
            print("x_distance = ", x_distance, ", y_distance = ", y_distance, ", z_distance = ", z_distance)

        else:
            continue
    else:
        continue

    cv2.imshow("Segmentation Results", annotated_frame)

    key = cv2.waitKey(2000)
    if key == 27 or key == 32: # Esc & Space
        print('break')
        break

    if  s != 1 and (x_distance == None or y_distance == None or z_distance == None):
        robot_move(Search = True)
        s = 1
    elif x_distance != None and y_distance != None and z_distance >= 100:
        robot_move(Object = True, X_distance = x_distance, Y_distance = y_distance, Z_distance = z_distance)
        robot_move(Treatment = True, Repeat = repeat, Width = width)
        x_distance, y_distance, z_distance, repeat, width = reset()
        s = 0
    else:
        print("Con't find the object.")
        time.sleep(1)

    if key == 27 or key == 32: # Esc & Space
        print('Break')
        break

robot_move(Home = True)