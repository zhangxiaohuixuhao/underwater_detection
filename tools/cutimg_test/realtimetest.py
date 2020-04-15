from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import glob
import cv2
import os
import csv
import time
'''
读取视频实时测试，一秒约两帧（3980*2160），GPU一块
'''
classes = ['holothurian', 'echinus', 'scallop', 'starfish']

config_file = '/DATA/zhanghui/underwater-objection-detection/configs/underwater/cas_r50/cascade_rcnn_r50_fpn_1x.py'
checkpoint_file = '/DATA/zhanghui/underwater-objection-detection/work_dirs/cascade_rcnn_r50_fpn_1x_original/latest.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

camera = cv2.VideoCapture('/DATA/zhanghui/underwater_data_handle/train/original/2160/end/video.mp4')

def saveresult(img, result, num):
    # for i in range(4):
    for j in range(len(result[0])):
        if float(result[0][j, 4]) > 0.5:
            cv2.rectangle(img, (int(result[0][j, 0]), int(result[0][j, 1])), (int(result[0][j, 2]), int(result[0][j, 3])), (0, 255, 0), 2)
            cv2.putText(img, classes[0] + '_' + str(round(result[0][j, 4], 2)), (int(result[0][j, 0] - 10), int(result[0][j, 1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (6, 230, 230), 2)
        else:
            continue
    cv2.imwrite('/DATA/zhanghui/underwater_data_handle/train/original/2160/videotemp/' + str(num) + '.jpg',
                img)

print('Press "Esc", "q" or "Q" to exit.')
num = 1
ret_val = True
while ret_val:
    start = time.time()
    ret_val, img = camera.read()
    # print(num)
    result = inference_detector(model, img)
    end = time.time()
    print(end - start)
    saveresult(img, result, num)
    num = num + 1
    ch = cv2.waitKey(1)
    if ch == 27 or ch == ord('q') or ch == ord('Q'):
        break
    show_result(img, result, model.CLASSES, score_thr=0.5, wait_time=1)
