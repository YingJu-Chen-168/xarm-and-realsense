import numpy as np

number_of_image = 62

num = np.random.permutation(number_of_image)
num = num + 1
print(num)
val_num = num[0 : int(number_of_image/10)]
print(val_num)
train_num = num[int(number_of_image/10) : number_of_image-1]
print(train_num)
test_num = num[number_of_image-1]
print(test_num)

train_path = 'C:/Ying-Ju Chen/Lab/robot xarm/GitHub/CO2Dnet/data/ImageSets/Main/train.txt'
test_path = 'C:/Ying-Ju Chen/Lab/robot xarm/GitHub/CO2Dnet/data/ImageSets/Main/test.txt'
val_path = 'C:/Ying-Ju Chen/Lab/robot xarm/GitHub/CO2Dnet/data/ImageSets/Main/val.txt'

with open(val_path, 'w') as f:
    for i in range(np.size(val_num)):
        f.write(f"{val_num[i]}\n")

with open(train_path, 'w') as f:
    for i in range(np.size(train_num)):
        f.write(f"{train_num[i]}\n")

with open(test_path, 'w') as f:
        f.write(f"{test_num}\n")