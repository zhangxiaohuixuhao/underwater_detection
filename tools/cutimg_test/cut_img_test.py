from mmdet.apis import init_detector, inference_detector, show_result
import glob
import os
import shutil
import cv2
import math
import numpy as np
import csv
'''
裁剪图像进行测试，并对结果进行按照面积大小进行NMS，保存结果为csv文件
'''
classes = ['holothurian', 'echinus', 'callop', 'starfish']
config_file = '/DATA/zhanghui/underwater-obj_de/configs/underwater/cas_x101/cascade_rcnn_x101_64x4d_fpn_1x.py'
checkpoint_file = '/DATA/zhanghui/underwater-obj_de/work_dirs/cas_x101_64x4d_fpn_htc_1x/latest.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

def NMS(dets, thresh):
    # 首先数据赋值和计算对应矩形框的面积
    # dets的数据格式是dets[[xmin,ymin,xmax,ymax,scores]....]
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 5]

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

def recover_result(result, area_point):
    for i in range(4):
        temp = result[i][:, 0]
        result[i][:, 0] = result[i][:, 0] + area_point[0]
        result[i][:, 1] = result[i][:, 1] + area_point[1]
        result[i][:, 2] = result[i][:, 2] + area_point[0]
        result[i][:, 3] = result[i][:, 3] + area_point[1]
    return result

def gather_lab(all_result, result):
    for i in range(4):
        result[i] = result[i][result[i][:, 4] > 0.1, :]
        if len(result[i]) > 0:
            all_result[i] = np.concatenate((all_result[i], result[i]), axis=0)
    return all_result

def sort_result(all_result):
    for i in range(4):
        temp = all_result[i]
        temp = temp[np.lexsort(-temp.T)]
        all_result[i] = temp
    return all_result

def cut_save_img(imgname):
    shutil.copy(imgname, '/DATA/zhanghui/underwater-obj_de/data/temp_path/' + os.path.basename(imgname))
    img = cv2.imread(imgname)
    w, h = img.shape[:2]
    all_result = []
    if w < 1080:
        temp = np.array([])
        temp.shape = 0, 5
        all_result.append(temp)
        all_result.append(temp)
        all_result.append(temp)
        all_result.append(temp)
        result = inference_detector(model, '/DATA/zhanghui/underwater-obj_de/data/temp_path/' + os.path.basename(imgname))
        all_result = gather_lab(all_result, result)
    else:
        temp = np.array([])
        temp.shape = 0, 5
        all_result.append(temp)
        all_result.append(temp)
        all_result.append(temp)
        all_result.append(temp)
        result = inference_detector(model, '/DATA/zhanghui/underwater-obj_de/data/temp_path/' + os.path.basename(imgname))
        all_result = gather_lab(all_result, result)
        for i in range(math.ceil(float(h) / 960.0)):
            for j in range(math.ceil(float(w) / 540.0)):
                area_point = []
                left_x = 960 * i
                left_y = 540 * j
                right_x = min((left_x + 960), h)
                right_y = min((left_y + 540), w)
                if right_y - left_y < 540:
                    left_y = right_y - 540
                if right_x - left_x < 960:
                    left_x = right_x - 960
                area_point.append(left_x)
                area_point.append(left_y)
                area_point.append(right_x)
                area_point.append(right_y)
                temp_img = img[left_y: right_y, left_x: right_x, :]
                temp_imgname = '/DATA/zhanghui/underwater-obj_de/data/temp_path/' + os.path.basename(imgname)[:-4] + \
                               '_' + str(left_x) + '_' + str(left_y) + '_' + str(right_x) + '_' + str(right_y) + '.jpg'
                cv2.imwrite(temp_imgname, temp_img)
                result = inference_detector(model, temp_imgname)
                result = recover_result(result, area_point)
                all_result = gather_lab(all_result, result)
        all_result = sort_result(all_result)
    return all_result

def save_csv(csv_writer, imgname, lab):
    for i in range(4):
        for j in range(len(lab[i])):
            csv_writer.writerow([str(classes[i]), str(os.path.basename(imgname)[:-4] + '.xml'), str(lab[i][j, 4]),
                                 str(int(lab[i][j, 0])), str(int(lab[i][j, 1])), str(int(lab[i][j, 2])), str(int(lab[i][j, 3]))])

def result_sort(result):
    for i in range(4):
        for j in range(len(result[i])):
            temp = np.zeros((len(result[i]), 6))
            temp[:, :5] = result[i]
            temp[:, 5] = (result[i][:, 2] - result[i][:, 0]) * (result[i][:, 3] - result[i][:, 1])
            keep = NMS(temp, 0.7)
            temp = temp[keep, :5]
            result[i] = temp
    return result



def main():
    fcsv = open('/DATA/zhanghui/underwater-obj_de/submit/test_yb.csv', 'w', encoding='utf-8')
    csv_writer = csv.writer(fcsv)
    csv_writer.writerow(["name", "image_id", "confidence", "xmin", "ymin", "xmax", "ymax"])
    imgpath = '/DATA/zhanghui/underwater-obj_de/data/test-A-image/'
    num = 0
    # for imgname in glob.glob(imgpath + '000003.jpg'):
    imgname = '/DATA/zhanghui/underwater-obj_de/data/test-A-image/000003.jpg'
    if not os.path.exists('/DATA/zhanghui/underwater-obj_de/data/temp_path'):
        os.mkdir('/DATA/zhanghui/underwater-obj_de/data/temp_path')
    else:
        files = glob.glob('/DATA/zhanghui/underwater-obj_de/data/temp_path/*.jpg')
        for f in files:
            os.remove(f)
    result = cut_save_img(imgname)
    end_result = result_sort(result)
    show_result(imgname, end_result, model.CLASSES)
    # save_csv(csv_writer, imgname, end_result)
    print(num)
    num = num + 1
    fcsv.close()


if __name__ == '__main__':
    main()