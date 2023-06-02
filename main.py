# encoding:utf-8
import urllib.request
import base64
import json
import os
import pycocotools.mask as mask_util
import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
easydl图像分割
'''

def main():

    path = os.path.join('C:\\Users\\sky\\Desktop\\feedport\\4_right\\标注\\8#(52-200).json')
    img_path = 'C:\\Users\\sky\\Desktop\\feedport\\1_right\\i\\'
    if os.path.isfile(path):
        data = json.load(open(path))
        images = data['images']
        annotations = data['annotations']

        image_dict = {}
        for item in annotations:
            imageid = item['image_id']
            if imageid not in image_dict:
                image_dict[imageid] = []  # 创建一个空列表
            image_dict[imageid].append(item)

        for imageid, items in image_dict.items():
            items = sorted(items, key=lambda x: x['category_id'])#正常用这个
            # items = sorted(items, key=lambda x: (x['category_id'] != 2, x['category_id']))#前后景标反用这个，且下面绘图交换1，2
            ann_img = np.zeros((images[imageid - 1]['height'], images[imageid - 1]['width'], 3)).astype('uint8')
            for item in items:
                points = []
                points2 = []
                for i in range(0, len(item['segmentation'][0]), 2):
                    x = item['segmentation'][0][i]
                    y = item['segmentation'][0][i + 1]
                    point = (x, y)
                    points.append(point)
                    if (item['category_id'] == 1):
                        cv2.fillPoly(ann_img, np.array([points], dtype=np.int32), (0, 0, 128))
                    elif(item['category_id'] == 2):
                        points2.append(point)
                        cv2.fillPoly(ann_img, np.array([points], dtype=np.int32), (0, 128, 0))

            cv2.imwrite(images[imageid - 1]['file_name'][:-4] + '.png', ann_img)


if __name__ == '__main__':
    main()

