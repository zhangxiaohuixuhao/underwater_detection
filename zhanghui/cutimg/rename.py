import cv2
import glob
import os
import shutil

img_path = '/DATA/zhanghui/underwater_data_handle/train/original/2160/end/'
lab_path = '/DATA/zhanghui/underwater_data_handle/train/original/test-B-lab/'
saveimg = '/DATA/zhanghui/underwater_data_handle/train/original/2160/videotemp/'
savelab = '/DATA/zhanghui/underwater_data_handle/train/original/temp_lab/'
# cur = 116
#
# for imgfile in glob.glob(img_path + '*.jpg'):
#     aa = int(os.path.basename(imgfile)[:-4])
#     shutil.copy(imgfile, saveimg + str(aa - 115) + '.jpg')
#     print('ok')

#
# for imgname in glob.glob(img_path + '*.jpg'):
#     numa = os.path.basename(imgname)[:-4].split('_')
#     numb = os.path.basename(imgname)[:-4].split('-')
#     if len(numa) > len(numb):
#         temp_img = saveimg + os.path.basename(imgname)
#     else:
#         temp_img = saveimg + numb[0] + '_' + numb[1] + '.jpg'
#     shutil.copy(imgname, temp_img)


# for imgname in glob.glob(img_path + '*.jpg'):
#     img = cv2.imread(imgname)
#     w, h = img.shape[:2]
#     if w == 1080:
#         # change_name = saveimg + str(cur) + '.jpg'
#         shutil.copy(imgname, saveimg + os.path.basename(imgname)




# listname = []
# for num in range(1, 245):
#     imgname = img_path + str(num) + '.jpg'
#     listname.append(os.path.basename(imgname)[:-4])
#     imgappend = []
#     img = glob.glob(img_path + os.path.basename(imgname)[:-4] + '_' +'*.jpg')
#     for i in range(len(img)):
#         imgappend.append(os.path.basename(img[i])[:-4])
#     imgappend.sort()
#     for i in range(len(imgappend)):
#         listname.append(imgappend[i])
# for endnum in range(len(listname)):
#     imgname = img_path + listname[endnum] + '.jpg'
#     shutil.copy(imgname, saveimg + str(endnum + 1) + '.jpg')
# print('ok')


num = 1
numlist = os.listdir(img_path)
numlist.sort()
for i in range(len(numlist)):
    imgname = saveimg + str(num) + '.jpg'
    shutil.copy(img_path + numlist[i], imgname)
    num = num + 1