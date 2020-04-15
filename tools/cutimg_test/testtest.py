from mmdet.apis import init_detector, inference_detector, show_result
import glob
import os
import shutil
import cv2
import math
import numpy as np
import csv

classes = ['holothurian', 'echinus', 'callop', 'starfish']
config_file = '/DATA/zhanghui/underwater-objection-detection/configs/underwater/cas_x101/cascade_rcnn_x101_64x4d_fpn_1x.py'
checkpoint_file = '/DATA/zhanghui/underwater-obj_de/workdirs/cas_x101_64x4d_fpn_htc_1x/latest.pth'
# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
print('ok')

def recover_result(result, area_point):
    for i in range(4):
        temp = result[i][:, 0]
        result[i][:, 0] = result[i][:, 0] + area_point[0]
        result[i][:, 1] = result[i][:, 1] + area_point[1]
        result[i][:, 2] = result[i][:, 2] + area_point[2]
        result[i][:, 3] = result[i][:, 3] + area_point[3]
    return result

def gather_lab(all_result, result):
    for i in range(4):
        all_result[i] = np.concatenate((all_result[i], result[i]), axis=0)
    return all_result

def sort_result(all_result):
    for i in range(4):
        temp = all_result[i]
        temp = temp[np.lexsort(-temp.T)]
        all_result[i] = temp
    return all_result

def cut_save_img(imgname):
    shutil.copy(imgname, '/DATA/zhanghui/underwater-objection-detection/data/temp_path/' + os.path.basename(imgname))
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
        result = inference_detector(model, '/DATA/zhanghui/underwater-objection-detection/data/temp_path/' + os.path.basename(imgname))
        all_result = gather_lab(all_result, result)
    else:
        temp = np.array([])
        temp.shape = 0, 5
        all_result.append(temp)
        all_result.append(temp)
        all_result.append(temp)
        all_result.append(temp)
        result = inference_detector(model, '/DATA/zhanghui/underwater-objection-detection/data/temp_path/' + os.path.basename(imgname))
        all_result = gather_lab(all_result, result)
        for i in range(math.ceil(float(h) / 860.0)):
            for j in range(math.ceil(float(w) / 440.0)):
                area_point = []
                left_x = 860 * i
                left_y = 440 * j
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
                temp_imgname = '/DATA/zhanghui/underwater-objection-detection/data/temp_path/' + os.path.basename(imgname)[:-4] + \
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
            if float(lab[i][j, 4]) > 0.05:
                csv_writer.writerow([str(classes[i]), str(os.path.basename(imgname)[:-4] + '.xml'), str(lab[i][j, 4]),
                                     str(int(lab[i][j, 0])), str(int(lab[i][j, 1])), str(int(lab[i][j, 2])), str(int(lab[i][j, 3]))])
            else:
                continue

def main():
    fcsv = open('/DATA/zhanghui/underwater-objection-detection/submit/test_yb.csv', 'w', encoding='utf-8')
    csv_writer = csv.writer(fcsv)
    csv_writer.writerow(["name", "image_id", "confidence", "xmin", "ymin", "xmax", "ymax"])
    imgpath = '/DATA/zhanghui/underwater-objection-detection/data/test-A-image/'
    for imgname in glob.glob(imgpath + '*.jpg'):

        if not os.path.exists('/DATA/zhanghui/underwater-objection-detection/data/temp_path'):
            os.mkdir('/DATA/zhanghui/underwater-objection-detection/data/temp_path')
        else:
            files = glob.glob('/DATA/zhanghui/underwater-objection-detection/data/temp_path/*.jpg')
            for f in files:
                os.remove(f)
        result = cut_save_img(imgname)
        save_csv(csv_writer, imgname, result)
    fcsv.close()


if __name__ == '__main__':
    main()