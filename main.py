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

def reset():
    x_distance = None
    y_distance = None
    z_distance = None
    repeat = None
    width = None
    return x_distance, y_distance, z_distance, repeat, width

def search():
    print("Go to Search")
    RobotSearch.pprint('xArm-Python-SDK Version:{}'.format(version.__version__))
    arm = XArmAPI('192.168.1.222', baud_checkset=False)
    robot_main = RobotSearch(arm)
    robot_main.run()

def detection(x, y, z):
    print("Go to Detection")
    RobotDetection.pprint('xArm-Python-SDK Version:{}'.format(version.__version__))
    arm = XArmAPI('192.168.1.222', baud_checkset=False)
    robot_main = RobotDetection(arm, x, y, z) 
    robot_main.run()

def treatment(repeat_time, width):
    print("Go to Treatment")
    RobotTreatment.pprint('xArm-Python-SDK Version:{}'.format(version.__version__))
    arm = XArmAPI('192.168.1.222', baud_checkset=False)
    robot_main = RobotTreatment(arm, repeat_time, width) 
    robot_main.run()

def gohome():
    print("Go Home")
    RobotGoHome.pprint('xArm-Python-SDK Version:{}'.format(version.__version__))
    arm = XArmAPI('192.168.1.222', baud_checkset=False)
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

W = 640 
H = 480 
x_distance, y_distance, z_distance, repeat, width = reset()
a = 0

config = rs.config()
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
pipeline = rs.pipeline()
profile = pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)

model_directory = 'yolov8n.pt'      #os.environ['HOME'] + '/yolov8_rs/yolov8m.pt'
model = YOLO(model_directory)

search()
time.sleep(2)

while True:

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)

    results = model(color_image)

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

            cv2.rectangle(depth_colormap, (x1, y1), (x2, y2), (211, 0, 148),
                          thickness = 2, lineType = cv2.LINE_4)
            cv2.putText(depth_colormap, text = model.names[clss], org = (x1 - 50, y2 + 20), fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale = 0.7, color = (211, 0, 148), thickness = 2, lineType = cv2.LINE_4)
            
            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    
            cv2.circle(depth_colormap, (x1, y1), 5, (0, 0, 255), -1)  # 紅色圓點
            cv2.circle(depth_colormap, (x2, y1), 5, (0, 255, 0), -1)  # 綠色圓點
            cv2.circle(depth_colormap, (x2, y2), 5, (255, 0, 0), -1)  # 藍色圓點

            # 確定座標值是否在depth_image的範圍之内
            depth1 = check_point(x1, y1)
            depth2 = check_point(x2, y1)
            depth3 = check_point(x2, y2)
            
            if depth1 != None and depth2 != None and depth3 != None:
                point1 = rs.rs2_deproject_pixel_to_point(intrinsics, [x1, y1], depth1)
                point2 = rs.rs2_deproject_pixel_to_point(intrinsics, [x2, y1], depth2)
                point3 = rs.rs2_deproject_pixel_to_point(intrinsics, [x2, y2], depth3)

                width = get_length(point2, point1)
                height = get_length(point3, point2)
                repeat = int(round(height/30))+1
                print("寬 is:", width, "mm. 長 is:", height, "mm.", "repeat = ", repeat)

                depth4 = check_point(int(W/2), int(H/2))
                point4 = rs.rs2_deproject_pixel_to_point(intrinsics, [int(W/2), int(H/2)], depth4)
                x_distance = int((point4[0] - point1[0]) *1000)
                y_distance = int((point4[1] - point1[1]) *1000)
                z_distance = int(depth4*1000)
                print("x_distance = ", x_distance, ", y_distance = ", y_distance, ", z_distance = ", z_distance)

                cv2.putText(depth_colormap, text = f'W: {str(width)}mm H: {str(height)}mm', org = (x1 - 50, y2 + 40), fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale = 0.7, color = (211, 0, 148), thickness = 2, lineType = cv2.LINE_4)

            else:
                continue

    annotated_frame = results[0].plot()
    images = np.hstack((annotated_frame, depth_colormap))
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', images)
    key = cv2.waitKey(3000)
    if key == 27 or key == 32: # Esc & Space
        print('break')
        break

    if  x_distance == None or y_distance == None or z_distance == None:
        search()
    elif x_distance != None and y_distance != None and z_distance >= 100:
        detection(y_distance - 30, x_distance, -(z_distance - 100))
        treatment(repeat, width)
    else:
        continue

    x_distance, y_distance, z_distance, repeat, width = reset()

    if key == 27 or key == 32: # Esc & Space
        print('break')
        break

gohome()