'''
根据change_biglab.py输出的类标文件进行按目标切图
思路：根据目标面积小于10000的目标在原图中进行切图，保留切完图中所有的目标坐标
'''
import cv2
import glob
import os
import numpy as np

alllab_txt = '/DATA/zhanghui/wanglei/train/changelab_data/temp_lab/' #全图类标文件
img_path = '/DATA/zhanghui/wanglei/train/cutdata/cut_temp_img/' #待裁剪图像全图
txt_path = '/DATA/zhanghui/wanglei/train/cutdata/cut_temp_lab/' #待裁剪目标的类标文件
saveimg_path = '/DATA/zhanghui/wanglei/train/cutdata/cut_img/' #裁剪之后图像保存路径
savetxt_path = '/DATA/zhanghui/wanglei/train/cutdata/cut_lab/' #裁剪图像整体的目标类标文件
cutimg_w = 540 #裁剪图像的高
cutimg_h = 960 #裁剪图像的宽

def lab_recover(w, h, lab):
    '''
    yolo标签信息：center_x, center_y, w, h 在原图中占有的比例
    还原真实坐标：center_x, center_y, w, h 
    '''
    labarray = []
    labarray.append(int(lab[0]))
    labarray.append(float(lab[1]) * h)
    labarray.append(float(lab[2]) * w)
    labarray.append(float(lab[3]) * h)
    labarray.append(float(lab[4]) * w)
    return labarray

def lab_recover_all(w, h, lab):
    '''
    yolo标签信息：center_x, center_y, w, h 在原图中占有的比例
    还原真实坐标：left_x, left_y, right_x, right_y 
    '''
    labarray = []
    labarray.append(int(lab[0]))
    center_x = float(lab[1]) * h
    center_y = float(lab[2]) * w
    lab_h = float(lab[3]) * h
    lab_w = float(lab[4]) * w
    labarray.append(int(center_x - lab_h / 2))
    labarray.append(int(center_y - lab_w / 2))
    labarray.append(int(center_x + lab_h / 2))
    labarray.append(int(center_y + lab_w / 2))
    return labarray

def cutimg_gt(w, h, lab):
    '''
    根据待裁剪目标位置裁剪其周围大小为cutimg_h * cutimg_w的图像
    返回待裁剪图像的在原图中的位置坐标
    '''
    cutimg_lab = []
    center_x = lab[1]
    center_y = lab[2]
    left_x = max((center_x - cutimg_h / 2), 0)
    left_y = max((center_y - cutimg_w / 2), 0)
    right_x = min((center_x + cutimg_h / 2), h)
    right_y = min((center_y + cutimg_w / 2), w)
    if left_x == 0 and (right_x - left_x) < cutimg_h:
        right_x = cutimg_h
    if left_y == 0 and (right_y - left_y) < cutimg_w:
        right_y = cutimg_w
    if right_x == h and (right_x - left_x) < cutimg_h:
        left_x = right_x - cutimg_h
    if right_y == w and (right_y - left_y) < cutimg_w:
        left_y = right_y - cutimg_w
    cutimg_lab.append(lab[0])
    cutimg_lab.append(left_x)
    cutimg_lab.append(left_y)
    cutimg_lab.append(right_x)
    cutimg_lab.append(right_y)
    cutimg_lab.append(right_x - left_x)
    cutimg_lab.append(right_y - left_y)
    return cutimg_lab

def right_crop(cutimg_lab, lab, w, h):
    '''
    :param cutimg_lab: 待裁剪图像的在原图中的位置坐标
    :param lab: 目标与裁剪图像有交集的目标在原图中的坐标位置
    :return crop_lab: 目标与裁剪图像交集区域在裁剪图像中的位置：center_x, center_y, w, h 在图中占的比例
    '''
    crop_lab = []
    center_x = float(lab[1]) * h
    center_y = float(lab[2]) * w
    w_crop = float(lab[3]) * h
    h_crop = float(lab[4]) * w
    left_x = center_x - w_crop / 2
    left_y = center_y - h_crop / 2
    right_x = center_x + w_crop / 2
    right_y = center_y + h_crop / 2
    if left_x < cutimg_lab[1] and right_x > cutimg_lab[1]:
        left_x = cutimg_lab[1]
    if left_x < cutimg_lab[3] and right_x > cutimg_lab[3]:
        right_x = cutimg_lab[3]
    if left_y < cutimg_lab[2] and right_y > cutimg_lab[2]:
        left_y = cutimg_lab[2]
    if left_y < cutimg_lab[4] and right_y > cutimg_lab[4]:
        right_y = cutimg_lab[4]
    w_little = right_x - left_x ##cutimg de zhenshi kuan gao
    h_little = right_y - left_y
    x_little = left_x + w_little / 2 - cutimg_lab[1]
    y_little = left_y + h_little / 2 - cutimg_lab[2]
    crop_center_x = x_little / cutimg_lab[5]
    crop_center_y = y_little / cutimg_lab[6]
    aa = w_little / cutimg_lab[5]
    bb = h_little / cutimg_lab[6]
    crop_lab.append(lab[0])
    crop_lab.append(crop_center_x)
    crop_lab.append(crop_center_y)
    crop_lab.append(aa)
    crop_lab.append(bb)
    return crop_lab

def march_or_not(cutimg_lab, other_lab):
    # # 两个检测框框是否有交叉，如果有交集则返回重叠度 IOU, 如果没有交集则返回 0
    # def bb_overlab(x1, y1, w1, h1, x2, y2, w2, h2):
    #     说明：图像中，从左往右是 x 轴（0~无穷大），从上往下是 y 轴（0~无穷大），从左往右是宽度 w ，从上往下是高度 h
    #     :param x1: 第一个框的左上角 x 坐标
    #     :param y1: 第一个框的左上角 y 坐标
    #     :param w1: 第一幅图中的检测框的宽度
    #     :param h1: 第一幅图中的检测框的高度
    #     :return: 两个如果有交集则返回重叠度 IOU, 如果没有交集则返回 0
    #     '''
    #     if(x1>x2+w2):
    #         return 0
    #     if(y1>y2+h2):
    #         return 0
    #     if(x1+w1<x2):
    #         return 0
    #     if(y1+h1<y2):
    #         return 0
    #     colInt = abs(min(x1 +w1 ,x2+w2) - max(x1, x2))
    #     rowInt = abs(min(y1 + h1, y2 +h2) - max(y1, y2))
    #     overlap_area = colInt * rowInt
    #     area1 = w1 * h1
    #     area2 = w2 * h2
    #     return overlap_area / (area1 + area2 - overlap_area)
    if (cutimg_lab[1] > other_lab[3]):
        return False
    if (cutimg_lab[2] > other_lab[4]):
        return False
    if (cutimg_lab[3] < other_lab[1]):
        return False
    if (cutimg_lab[4] < other_lab[2]):
        return False
    colInt = abs(min(cutimg_lab[3], other_lab[3]) - max(cutimg_lab[1], other_lab[1]))
    rowInt = abs(min(cutimg_lab[4], other_lab[4]) - max(cutimg_lab[2], other_lab[2]))
    overlap_area = colInt * rowInt
    if overlap_area > 0:
        return True
    else:
        return False



def cutimg(img, lab, lab_all):
    '''
    :param img：全图
    :param lab：全部待裁剪目标坐标位置
    :param lab_all：全图目标坐标位置
    :return first_img：根据第一个待裁剪目标进行裁剪的图像
    :return left_lab：经过判断全部待裁剪目标中心是否在已裁剪的图像中，剩余没有在该裁剪图像中的待裁剪目标
    :return write_lab：在该裁剪图像中的所有目标的坐标位置信息
    '''
    left_lab = []
    write_lab = []
    w, h = img.shape[:2]
    first_lab = lab_recover(w, h, lab[0])
    cutimg_lab = cutimg_gt(w, h, first_lab)
    first_img = img[int(cutimg_lab[2]): int(cutimg_lab[4]), int(cutimg_lab[1]): int(cutimg_lab[3]), :]
    # write_lab.append(lab[0])
    for i in range(len(lab)):
        other_lab = lab_recover(w, h, lab[i])
        # march = march_or_not(cutimg_lab, other_lab)
        if other_lab[1] < cutimg_lab[1] or other_lab[2] < cutimg_lab[2] or other_lab[1] > cutimg_lab[3] or other_lab[2] > cutimg_lab[4]:
            left_lab.append(lab[i])
    for i in range(len(lab_all)):
        other_lab = lab_recover_all(w, h, lab_all[i])
        march = march_or_not(cutimg_lab, other_lab)
        if march:
            write_lab.append(right_crop(cutimg_lab, lab_all[i], w, h))
    return first_img, left_lab, write_lab

def txtarray(txtname):
    '''
    将TXT类标转化成为array
    '''
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
    '''
    将裁剪图像中所有的目标位置写入TXT文件中
    '''
    for i in write_lab:
        for j in range(5):
            f.write(str(i[j]) + ' ')
        f.write('\r\n')
    f.close()

def main():
    for txtname in glob.glob(txt_path + '*.txt'):
        imgname = img_path + os.path.basename(txtname)[:-4] + '.jpg'
        txtlab = alllab_txt + os.path.basename(txtname)
        print(os.path.basename(txtname)[:-4] + '.jpg')
        img = cv2.imread(imgname)
        w, h = img.shape[:2]
        lab_array = txtarray(txtname) #待裁剪类标
        all_lab = txtarray(txtlab) #带裁剪目标所在图像的类标文件
        if len(lab_array) > 0:
            #按照待裁剪类标进行目标裁剪原图，得到裁剪图像first_img，若有待裁剪目标出现在first_img中，则不再以该待裁剪目标进行裁剪图像
            first_img, left_lab, write_lab = cutimg(img, lab_array, all_lab) 
            num = 1
            f = open(savetxt_path + os.path.basename(txtname)[:-4] + '_' + str(num) + '.txt', 'w')
            writetxt(f, write_lab)
            cv2.imwrite(saveimg_path + os.path.basename(txtname)[:-4] + '_' + str(num) + '.jpg', first_img)
            while len(left_lab) > 0:
                num = num + 1
                first_img, left_lab, write_lab = cutimg(img, left_lab, all_lab)
                f = open(savetxt_path + os.path.basename(txtname)[:-4] + '_' + str(num) + '.txt', 'w')
                writetxt(f, write_lab)
                cv2.imwrite(saveimg_path + os.path.basename(txtname)[:-4] + '_' + str(num) + '.jpg', first_img)
        print('ok')

if __name__ == '__main__':
    main()