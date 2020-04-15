import pandas as pd
import csv
import glob
import cv2
import numpy as np
import os

'''
根据得到的CVS文件，判断置信度是否大于0.3，将大于0.3的保存成一个新的csv文件，图像的可视化，并将新文件的生成对应图片的yolo标签TXT文件。
'''
def readCSV2List(filePath):
    try:
        file = open(filePath, 'r', encoding="gbk")# 读取以utf-8
        context = file.read() # 读取成str
        list_result = context.split("\n")#  以回车符\n分割成单独的行
        #每一行的各个元素是以【,】分割的，因此可以
        length = len(list_result)
        for i in range(length):
            list_result[i] = list_result[i].split(",")
        return list_result
    except Exception :
        print("文件读取转换失败，请检查文件路径及文件编码是否正确")
    finally:
        file.close()# 操作完成一定要关闭


def confdence_change(csvpath):
    data = pd.read_csv(csvpath)
    data_new = data[data['confidence'] > 0.3]#置信度设置为0.3
    data_new.to_csv('test.csv', index=0)
    list_test = readCSV2List('test.csv')[1:]#返回值为一个list
    return list_test

# def see_img_lab(imgpath, all_lab, save_imgpath):
##根据返回的list进行图像的可视化
#     for imgname in glob.glob(imgpath + '*.jpg'):
#         img = cv2.imread(imgname)
#         temp_xml = os.path.basename(imgname)[:-4] + '.xml'
#         img_lab = all_lab[all_lab[:, 1] == temp_xml, :]
#
#         for j in range(len(img_lab)):
#             cv2.rectangle(img, (int(img_lab[j, 3]), int(img_lab[j, 4])),
#                           (int(img_lab[j, 5]), int(img_lab[j, 6])), (255, 0, 0), 2)
#             cv2.putText(img, str(img_lab[j, 0]) + '_' + str(round(float(img_lab[j, 2]), 5)),
#                         (int(img_lab[j, 3]) - 10, int(img_lab[j, 4]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (6, 230, 230), 2)
#         cv2.imwrite(save_imgpath + os.path.basename(imgname), img)


def writetxt(f, write_lab):
    #将结果写入TXT文件中
    for i in write_lab:
        for j in range(5):
            f.write(str(i[j]) + ' ')
        f.write('\r\n')
    f.close()

def change_lab(img_lab, w, h):
    #将结果转换成为yolo标签的格式
    classes = ['holothurian', 'echinus', 'scallop', 'starfish']
    temp_lab = np.array([])
    temp_lab.shape = 0, 5
    for i in range(len(img_lab)):
        temp = np.zeros((1, 5))
        temp[:, 0] = int(classes.index(img_lab[i, 0]))
        labh = int(img_lab[i, 5]) - int(img_lab[i, 3])
        labw = int(img_lab[i, 6]) - int(img_lab[i, 4])
        temp[:, 1] = (int(img_lab[i, 3]) + labh / 2) / h
        temp[:, 2] = (int(img_lab[i, 4]) + labw / 2) / w
        temp[:, 3] = labh / h
        temp[:, 4] = labw / w
        temp_lab = np.concatenate((temp_lab, temp), axis=0)
    return temp_lab


def see_img_lab(imgpath, all_lab, save_imgpath):
    #将设置置信度之后的标签分离并按照图像名称分别保存为txt文件。
    for imgname in glob.glob(imgpath + '*.jpg'):
        print(imgname)
        img = cv2.imread(imgname)
        w, h = img.shape[:2]
        temp_xml = os.path.basename(imgname)[:-4] + '.xml'
        img_lab = all_lab[all_lab[:, 1] == temp_xml, :]
        img_lab = change_lab(img_lab, w, h)
        f = open(save_imgpath + os.path.basename(imgname)[:-4] + '.txt', 'w')
        writetxt(f, img_lab)



def main():
    csvpath = '/DATA/zhanghui/underwater-obj_de/submit/testB.csv'
    imgpath = '/DATA/zhanghui/underwater-obj_de/test-B-image/'
    save_imgpath = '/DATA/zhanghui/underwater-obj_de/change_txt/'
    list_test = confdence_change(csvpath)#进行csv文件的转换，包括设置置信度，并将csv文件读取成为一个list
    all_lab = np.array([])
    all_lab.shape = 0, 7
    list_test = list_test[:(len(list_test) - 1)]
    for i in range(len(list_test)):
        print(i)
        all_lab = np.concatenate((all_lab, np.array(list_test[i]).reshape(1, 7)), axis=0)
    see_img_lab(imgpath, all_lab, save_imgpath)


if __name__ == '__main__':
    main()
    print('ok')
