import cv2
import glob
import math
import numpy as np
from detector import *
import os
import csv

lab_name = ['holothurian', 'echinus', 'callop', 'starfish', 'waterweeds']
thresh = 0.5

def NMS(dets, thresh):
    # 首先数据赋值和计算对应矩形框的面积
    # dets的数据格式是dets[[xmin,ymin,xmax,ymax,scores]....]
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]

    # 这边的keep用于存放，NMS后剩余的方框
    keep = []
    # 取出分数从大到小排列的索引。.argsort()是从小到大排列，[::-1]是列表头和尾颠倒一下。
    index = scores.argsort()[::-1]
    # 上面这两句比如分数[0.72 0.8  0.92 0.72 0.81 0.9 ]
    #  对应的索引index[  2   5    4     1    3   0  ]记住是取出索引，scores列表没变。

    # index会剔除遍历过的方框，和合并过的方框。
    while index.size > 0:
        # print(index.size)
        # 取出第一个方框进行和其他方框比对，看有没有可以合并的
        i = index[0]  # every time the first is the biggst, and add it directly
        # 因为我们这边分数已经按从大到小排列了。
        # 所以如果有合并存在，也是保留分数最高的这个，也就是我们现在那个这个
        # keep保留的是索引值，不是具体的分数。
        keep.append(i)
        # 计算交集的左上角和右下角
        # 这里要注意，比如x1[i]这个方框的左上角x和所有其他的方框的左上角x的
        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        # 这边要注意，如果两个方框相交，X22-X11和Y22-Y11是正的。
        # 如果两个方框不相交，X22-X11和Y22-Y11是负的，我们把不相交的W和H设为0.
        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)

        # 计算重叠面积就是上面说的交集面积。不相交因为W和H都是0，所以不相交面积为0
        overlaps = w * h
        # 这个就是IOU公式（交并比）。
        # 得出来的ious是一个列表，里面拥有当前方框和其他所有方框的IOU结果。
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        # 接下来是合并重叠度最大的方框，也就是合并ious中值大于thresh的方框
        # 我们合并的操作就是把他们剔除，因为我们合并这些方框只保留下分数最高的。
        # 我们经过排序当前我们操作的方框就是分数最高的，所以我们剔除其他和当前重叠度最高的方框
        # 这里np.where(ious<=thresh)[0]是一个固定写法。
        idx = np.where(ious <= thresh)[0]
        # 把留下来框在进行NMS操作
        # 这边留下的框是去除当前操作的框，和当前操作的框重叠度大于thresh的框
        # 每一次都会先去除当前操作框，所以索引的列表就会向前移动移位，要还原就+1，向后移动一位
        index = index[idx + 1]  # because index start from 1
    return keep

def change_NMS(all_lab, thresh):
    keep_lab = []
    for i in range(len(lab_name)):
        temp_lab = []
        for j in range(len(all_lab)):
            temp = []
            if all_lab[j][0] == lab_name[i]:
                temp.append(int(all_lab[j][2]))
                temp.append(int(all_lab[j][3]))
                temp.append(int(all_lab[j][4]))
                temp.append(int(all_lab[j][5]))
                temp.append(float(all_lab[j][1]))
                temp_lab.append(temp)
        if len(temp_lab) > 0:
            aaa = np.array(temp_lab)
            keep = NMS(np.array(temp_lab), thresh)
            for n in range(len(keep)):
                keep_lab.append(all_lab[keep[n]])
    return keep_lab

def recover_lab(lab, area_point):
    label = []
    for i in range(len(lab)):
        temp_lab = []
        temp_lab.append(str(lab[i][0], encoding='utf-8'))
        temp_lab.append(round(lab[i][1], 5))
        # print(lab[i][2][0])
        left_x = max(int(lab[i][2][0] + area_point[0] - lab[i][2][2] / 2), 0)
        left_y = max(int(lab[i][2][1] + area_point[1] - lab[i][2][3] / 2), 0)
        right_x = int(lab[i][2][0] + area_point[0] + lab[i][2][2] / 2)
        right_y = int(lab[i][2][1] + area_point[1] + lab[i][2][3] / 2)
        temp_lab.append(left_x)
        temp_lab.append(left_y)
        temp_lab.append(right_x)
        temp_lab.append(right_y)
        label.append(temp_lab)
    return label



def recover_alllab(lab):
    label = []
    for i in range(len(lab)):
        temp_lab = []
        temp_lab.append(str(lab[i][0], encoding='utf-8'))
        temp_lab.append(round(lab[i][1], 5))
        left_x = max(int(lab[i][2][0] - lab[i][2][2] / 2), 0)
        left_y = max(int(lab[i][2][1] - lab[i][2][3] / 2), 0)
        right_x = int(lab[i][2][0] + lab[i][2][2] / 2)
        right_y = int(lab[i][2][1] + lab[i][2][3] / 2)
        temp_lab.append(left_x)
        temp_lab.append(left_y)
        temp_lab.append(right_x)
        temp_lab.append(right_y)
        label.append(temp_lab)
    return label

def save_img(img, all_lab, image_name):
    for i in range(len(all_lab)):
        left_x = int(all_lab[i][2])
        left_y = int(all_lab[i][3])
        right_x = int(all_lab[i][4])
        right_y = int(all_lab[i][5])
        cv2.rectangle(img, (left_x, left_y), (right_x, right_y), (0, 255, 0), 2)
        # cv2.putText(img, all_lab[i][0] + ' ' + str(all_lab[i][1]), (left_x-10, left_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (6, 230, 230), 2)
    cv2.imwrite('/DATA/zhanghui/wanglei/result/' + os.path.basename(image_name), img)
    # cv2.imshow('src', img)
    # cv2.waitKey(0)

def write_csv(csv_writer, imgname, lab):
    for i in range(len(lab)):
        if lab[i][0] == 'callop':
            lab[i][0] = 'scallop'
            csv_writer.writerow([str(lab[i][0]), str(os.path.basename(imgname)[:-4] + '.xml'), str(lab[i][1]), str(lab[i][2]), str(lab[i][3]), str(lab[i][4]), str(lab[i][5])])
        else:
            csv_writer.writerow([str(lab[i][0]), str(os.path.basename(imgname)[:-4] + '.xml'), str(lab[i][1]), str(lab[i][2]), str(lab[i][3]), str(lab[i][4]), str(lab[i][5])])
def main():
    # imgpath = '/DATA/zhanghui/wanglei/saveimg/000001.jpg'
    f = open('test_yb.csv', 'w', encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["name", "image_id", "confidence", "xmin", "ymin", "xmax", "ymax"])
    # for imgpath in glob.glob('/DATA/zhanghui/wanglei/test-A-image/*jpg'):
    for i in range(1, 801):
        print(i)
        imgpath = '/DATA/zhanghui/underwater-obj_de/data/test-A-image/' + str(("%06d" % (i))) + '.jpg'
        img = cv2.imread(imgpath)
        w, h = img.shape[:2]
        if w > 1080:
            num = 0
            all_lab = []
            for i in range(math.ceil(float(h) / 716.0)):
                for j in range(math.ceil(float(w) / 716.0)):
                    area_point = []
                    left_x = 716 * i
                    left_y = 716 * j
                    right_x = min((left_x + 816), h)
                    right_y = min((left_y + 816), w)
                    if (right_x - left_x) < 416:
                        left_x = right_x - 416
                    if (right_y - left_y) < 416:
                        left_y = right_y - 416
                    area_point.append(left_x)
                    area_point.append(left_y)
                    area_point.append(right_x)
                    area_point.append(right_y)
                    cut_img = img[left_y: right_y, left_x: right_x, :]
                    test_image = nparray_to_image(cut_img)
                    lab = detect(net, meta, test_image)
                    org_lab = recover_lab(lab, area_point)
                    if len(org_lab) > 0:
                        for n in range(len(org_lab)):
                            all_lab.append(org_lab[n])
            keep_lab = change_NMS(all_lab, thresh)
            write_csv(csv_writer, imgpath, keep_lab)
            # save_img(img, keep_lab, imgpath)
        else:
            test_image = nparray_to_image(img)
            lab = detect(net, meta, test_image)
            org_lab = recover_alllab(lab)
            write_csv(csv_writer, imgpath, org_lab)
            # save_img(img, org_lab, imgpath)
    f.close()
    print('ok')


if __name__ == '__main__':
    main()
