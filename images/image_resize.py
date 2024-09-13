import cv2

for i in range(11):
    img = cv2.imread(f"C:/Ying-Ju Chen/Lab/robot xarm/GitHub/curebot/images/temporary/{str(i+137)}.jpg")
    img = cv2.resize(img, (640, 640), interpolation = cv2.INTER_LINEAR)
    cv2.imwrite(f"C:/Ying-Ju Chen/Lab/robot xarm/GitHub/curebot/images/640/{str(i+137)}.jpg", img)