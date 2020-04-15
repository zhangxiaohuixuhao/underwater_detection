# coding=utf-8
'''
check label and delete null label file
according new label file check image
'''
import xml.etree.ElementTree as et
import os
import glob
import shutil
'''
判断XML文件中的类标，并按照生成的新的XML文件进行图像拷贝
'''
def change_car(path0, path_car, class_name):
    '''
    according label name check label file and delete null file
    '''
    xml_lst = os.listdir(path0)
    for axml in xml_lst:
        path_xml = os.path.join(path0, axml)
        tree = et.parse(path_xml)
        root = tree.getroot()
        # for child in root.findall('object'):
        #     name = child.find('name').text
        #     # # if name != 'holothurian':
        #     # if name not in class_name:
        #     #     root.remove(child)
        #     #     print('ok')
        num_child = root.findall('object')
        if len(num_child) == 0:
            print(path_xml)
            continue
        else:
            tree.write(os.path.join(path_car, axml))

def copyimg():
    '''
    according new label file check image
    '''
    path1 = '/DATA/zhanghui/wanglei/zhanghui/changelab_data/box/'
    path2 = '/DATA/lwang/data_sea/detectron_data/JPEGImages/'
    pathsave = '/DATA/zhanghui/wanglei/zhanghui/changelab_data/image/'
    file = glob.glob(path1 + '*.xml')
    for xmlpath in file:
        imgname = os.path.basename(xmlpath)[:-4] + '.jpg'
        imgpath = os.path.join(path2 + imgname)
        shutil.copy(imgpath, os.path.join(pathsave + imgname))

def main():
    path0 = '/DATA/zhanghui/underwater_data_handle/star_1/'
    path_car = '/DATA/zhanghui/underwater_data_handle/star/'
    class_name = ['holothurian', 'echinus', 'scallop', 'starfish']
    change_car(path0, path_car, class_name)
    # copyimg()

if __name__ == '__main__':
    main()