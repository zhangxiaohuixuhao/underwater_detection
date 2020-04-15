'''
根据目标框的大小选取目标框面积小于10000的目标
'''
import cv2
import glob
import os
import numpy as np
import csv
import matplotlib.pyplot as plt
import shutil

classes = ['holothurian', 'echinus', 'scallop', 'starfish'] #待选取目标的类别
img_path = '/DATA/zhanghui/wanglei/train/changelab_data/image/' #待选择图像的文件夹路径
txt_path = '/DATA/zhanghui/wanglei/train/changelab_data/temp_lab/' #标签路径，yolov3的标签格式
saveimg_path = '/DATA/zhanghui/wanglei/train/changelab_data/image/' #待选择图像的文件夹路径
savetxt_path = '/DATA/zhanghui/wanglei/train/changelab_data/temp_lab/' #标签路径，yolov3的标签格式

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

def writetxt(f, write_lab):
    for i in write_lab:
        for j in range(5):
            f.write(str(i[j]) + ' ')
        f.write('\r\n')
    f.close()

def main():
    num = 0
    for txtname in glob.glob(txt_path + '*.txt'):
        num = num + 1
        imgname = img_path + os.path.basename(txtname)[:-4] + '.jpg'
        print(os.path.basename(txtname)[:-4] + '.jpg')
        img = cv2.imread(imgname)
        w, h = img.shape[:2]
        if w >= 1080:
            print(w, h)
            lab_array = txtarray(txtname)
            writelab = []
            littlelab = []
            for i in range(len(lab_array)):
                lab = lab_recover(w, h, lab_array[i])
                area = lab[3] * lab[4]
                if area > 100 and area <= 10000:
                    littlelab.append(lab_array[i])
            if len(littlelab) > 0:
                f = open(savetxt_path + os.path.basename(txtname), 'w')
                writetxt(f, littlelab)
                shutil.copy(img_path + os.path.basename(txtname)[:-4] + '.jpg', saveimg_path + os.path.basename(txtname)[:-4] + '.jpg')
                # cv2.imwrite('/DATA/zhanghui/wanglei/train/cutdata/delet_img/' + os.path.basename(txtname)[:-4] + '.jpg', img)
    print(num)

if __name__ == "__main__":
    main()