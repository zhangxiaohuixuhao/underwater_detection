import cv2
import glob
import os
import numpy as np
import csv
import matplotlib.pyplot as plt
'''
判断每一类的目标大小分布
'''
img_path = '/DATA/zhanghui/wanglei/train/image/'
txt_path = '/DATA/zhanghui/wanglei/train/txtbox/'
saveimg_path = '/DATA/zhanghui/wanglei/testimg/'
classes = ['holothurian', 'echinus', 'scallop', 'starfish'] #haishen haidan shanbei haixing

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
area_hol = []
area_ech = []
area_sca = []
area_star = []
for txtname in glob.glob('/DATA/zhanghui/wanglei/train/txtbox/*.txt'):
    imgname = img_path + os.path.basename(txtname)[:-4] + '.jpg'
    print(os.path.basename(txtname)[:-4] + '.jpg')
    img = cv2.imread(imgname)
    w, h = img.shape[:2]
    lab_array = txtarray(txtname)
    for i in range(len(lab_array)):
        lab = lab_recover(w, h, lab_array[i])
        area = int(lab[4] * lab[3])
        if lab[0] == 0:
            lab = lab[1:]
            if area > 600000:
                cv2.rectangle(img, (int(lab[0] - lab[2] / 2), int(lab[1] - lab[3] / 2)),
                              (int(lab[0] + lab[2] / 2), int(lab[1] + lab[3] / 2)), (255, 0, 0), 2)
                cv2.putText(img, 'holo', (int(lab[0] - lab[2] / 2)-10, int(lab[1] - lab[3] / 2)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (6, 230, 230), 2)
                cv2.imwrite(saveimg_path + os.path.basename(txtname)[:-4] + '.jpg', img)
            # area_hol.append(area)
        if lab[0] == 1:
            lab = lab[1:]
            if area > 750000:
                cv2.rectangle(img, (int(lab[0] - lab[2] / 2), int(lab[1] - lab[3] / 2)),
                              (int(lab[0] + lab[2] / 2), int(lab[1] + lab[3] / 2)), (255, 0, 0), 2)
                cv2.putText(img, 'ech', (int(lab[0] - lab[2] / 2)-10, int(lab[1] - lab[3] / 2)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (6, 230, 230), 2)
                cv2.imwrite(saveimg_path + os.path.basename(txtname)[:-4] + '.jpg', img)
            # area_ech.append(area)
        if lab[0] == 2:
            lab = lab[1:]
            if area > 400000:
                cv2.rectangle(img, (int(lab[0] - lab[2] / 2), int(lab[1] - lab[3] / 2)),
                              (int(lab[0] + lab[2] / 2), int(lab[1] + lab[3] / 2)), (255, 0, 0), 2)
                cv2.putText(img, 'scal', (int(lab[0] - lab[2] / 2)-10, int(lab[1] - lab[3] / 2)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (6, 230, 230), 2)
                cv2.imwrite(saveimg_path + os.path.basename(txtname)[:-4] + '.jpg', img)
            # area_sca.append(area)
        if lab[0] == 3:
            lab = lab[1:]
            if area > 800000:
                cv2.rectangle(img, (int(lab[0] - lab[2] / 2), int(lab[1] - lab[3] / 2)),
                              (int(lab[0] + lab[2] / 2), int(lab[1] + lab[3] / 2)), (255, 0, 0), 2)
                cv2.putText(img, 'star', (int(lab[0] - lab[2] / 2)-10, int(lab[1] - lab[3] / 2)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (6, 230, 230), 2)
                cv2.imwrite(saveimg_path + os.path.basename(txtname)[:-4] + '.jpg', img)
            # area_star.append(area)

        # cv2.imshow('src', img)
        # cv2.waitKey(0)
        # print('ok')


        # area = int(lab[4] * lab[3])
        # if lab[0] == 0:
        #     area_hol.append(area)
        # if lab[0] == 1:
        #     area_ech.append(area)
        # if lab[0] == 2:
        #     area_sca.append(area)
        # if lab[0] == 3:
        #     area_star.append(area)
# print(len(area_hol))
# plt.hist(x=area_hol, bins=100000, color='steelblue', edgecolor='black')
# plt.xlabel('area')
# plt.ylabel('num')
# plt.title('holothurian' + '_' + str(len(area_hol)))
# plt.savefig('holothurian.jpg')

# plt.hist(x=area_ech, bins=100000, color='steelblue', edgecolor='black')
# plt.xlabel('area')
# plt.ylabel('num')
# plt.title('echinus' + '_' + str(len(area_ech)))
# plt.savefig('echinus.jpg')

# plt.hist(x=area_sca, bins=100000, color='steelblue', edgecolor='black')
# plt.xlabel('area')
# plt.ylabel('num')
# plt.title('scallop' + '_' + str(len(area_sca)))
# plt.savefig('scallop.jpg')
#
# plt.hist(x=area_star, bins=100000, color='steelblue', edgecolor='black')
# plt.xlabel('area')
# plt.ylabel('num')
# plt.title('starfish' + '_' + str(len(area_star)))
# plt.savefig('starfish.jpg')

        # lab = lab[1:]
    #     cv2.rectangle(img, (int(lab[0] - lab[2] / 2), int(lab[1] - lab[3] / 2)), (int(lab[0] + lab[2] / 2), int(lab[1] + lab[3] / 2)), (255, 0, 0), 2)
    # cv2.imwrite(saveimg_path + os.path.basename(txtname)[:-4] + '.jpg', img)
    # cv2.imshow('src', img)
    # cv2.waitKey(0)
    # print('ok')