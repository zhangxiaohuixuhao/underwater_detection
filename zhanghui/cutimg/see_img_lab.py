'''
根据yolo类标信息进行图像可视化
'''
import cv2
import glob
import os
import numpy as np
img_path = '/DATA/zhanghui/wanglei/train/cutdata/cut_img/'
txt_path = '/DATA/zhanghui/wanglei/train/cutdata/cut_lab/'
saveimg_path = '/DATA/zhanghui/wanglei/testimg/'


def lab_recover(w, h, lab):
    labarray = []
    labarray.append(int(lab[0]))
    labarray.append(float(lab[1]) * h)
    labarray.append(float(lab[2]) * w)
    labarray.append(float(lab[3]) * h)
    labarray.append(float(lab[4]) * w)
    return labarray

def txtarray(txtname):
    dataarray = []
    f = open(txtname).read().strip().split()
    temparray = []
    num = 1
    for line in f:
        temparray.append(line)
        num = num + 1
        if num == 6:
            dataarray.append(temparray)
            temparray = []
            num = 1
    return np.array(dataarray)


for txtname in glob.glob(txt_path + '*.txt'):
    imgname = img_path + os.path.basename(txtname)[:-4] + '.jpg'
    print(os.path.basename(txtname)[:-4] + '.jpg')
    img = cv2.imread(imgname)
    w, h = img.shape[:2]
    lab_array = txtarray(txtname)
    for i in range(len(lab_array)):
        lab = lab_recover(w, h, lab_array[i])
        lab = lab[1:]
        area = lab[2] * lab[3]
        cv2.rectangle(img, (int(lab[0] - lab[2] / 2), int(lab[1] - lab[3] / 2)), (int(lab[0] + lab[2] / 2), int(lab[1] + lab[3] / 2)), (255, 0, 0), 2)
        cv2.putText(img, str(int(area)), (int(lab[0] - lab[2] / 2) - 10, int(lab[1] - lab[3] / 2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (6, 230, 230), 1)
    cv2.imwrite(saveimg_path + os.path.basename(txtname)[:-4] + '.jpg', img)
    # cv2.imshow('src', img)
    # cv2.waitKey(0)
    # print('ok')