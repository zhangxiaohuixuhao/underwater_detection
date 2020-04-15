import cv2
import glob
import os
import shutil

img_path = '/DATA/zhanghui/wanglei/JPEGImages/'
lab_path = '/DATA/zhanghui/wanglei/labels/'
saveimg = '/DATA/zhanghui/wanglei/tempimg/'
savelab = '/DATA/zhanghui/wanglei/templab/'

for imgfile in glob.glob(img_path + '*.jpg'):
    img = cv2.imread(imgfile)
    w, h = img.shape[:2]
    if w < 1080:
        cv2.imwrite(saveimg + os.path.basename(imgfile), img)
        shutil.copy(lab_path + os.path.basename(imgfile)[:-4] + '.txt', savelab + os.path.basename(imgfile)[:-4] + '.txt')