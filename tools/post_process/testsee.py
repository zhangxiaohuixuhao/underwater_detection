import json
import os
import argparse
import cv2
underwater_classes = ['holothurian', 'echinus', 'scallop', 'starfish']
def parse_args():
    parser = argparse.ArgumentParser(description='json2submit_nms')
    parser.add_argument('--test_json', help='test result json', type=str)
    parser.add_argument('--submit_file', help='submit_file_name', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    test_json_raw = json.load(open("/DATA/zhanghui/underwater-objection-detection/data/train/annotations/testA.json", "r"))
    test_json = json.load(open("/DATA/zhanghui/underwater-objection-detection/results/" + 'test.json', "r"))
    submit_file_name = 'test.json'
    submit_path = '/DATA/zhanghui/underwater-objection-detection/submit/'
    os.makedirs(submit_path, exist_ok=True)
    img = test_json_raw['images']
    images = []
    csv_file = open(submit_path + submit_file_name, 'w')
    csv_file.write("name,image_id,confidence,xmin,ymin,xmax,ymax\n")
    imgid2anno = {}
    imgid2name = {}
    for imageinfo in test_json_raw['images']:
        imgid = imageinfo['id']
        imgid2name[imgid] = imageinfo['file_name']
    for anno in test_json:
        img_id = anno['image_id']
        if img_id not in imgid2anno:
            imgid2anno[img_id] = []
        imgid2anno[img_id].append(anno)
    for imgid, annos in imgid2anno.items():
        image_name = imgid2name[imgid]
        img = cv2.imread('/DATA/zhanghui/underwater-objection-detection/data/train/image/' + image_name)
        for anno in annos:
            xmin, ymin, w, h = anno['bbox']
            xmax = xmin + w
            ymax = ymin + h
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            confidence = anno['score']
            class_id = int(anno['category_id'])
            class_name = underwater_classes[class_id-1]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            image_id = image_name.split('.')[0] + '.xml'
            csv_file.write(class_name + ',' + image_id + ',' + str(confidence) + ',' + str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + '\n')
        cv2.imwrite('/DATA/zhanghui/underwater-objection-detection/data/testimg/' + image_name, img)
    csv_file.close()