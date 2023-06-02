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



        '''5_1_194_0标注了三个前景'''
        # annotation1 = annotations[30]
        # image_id1 = annotation1['image_id']
        # segmentation1 = annotation1['segmentation']
        # ann_img = np.zeros((images[image_id1 - 1]['height'], images[image_id1 - 1]['width'], 3)).astype('uint8')
        # points = []
        # for i in range(0, len(segmentation1[0]), 2):
        #     x = segmentation1[0][i]
        #     y = segmentation1[0][i + 1]
        #     point = (x, y)
        #     points.append(point)
        #
        # cv2.fillPoly(ann_img, np.array([points], dtype=np.int32), (0, 0, 128))
        # # cv2.imshow('Image', ann_img)
        # # cv2.waitKey(0)
        # # cv2.destroyAllWindows()
        #
        # annotation1 = annotations[31]
        # image_id1 = annotation1['image_id']
        # segmentation1 = annotation1['segmentation']
        # # ann_img = np.zeros((images[image_id1 - 1]['height'], images[image_id1 - 1]['width'], 3)).astype('uint8')
        # points = []
        # for i in range(0, len(segmentation1[0]), 2):
        #     x = segmentation1[0][i]
        #     y = segmentation1[0][i + 1]
        #     point = (x, y)
        #     points.append(point)
        #
        # cv2.fillPoly(ann_img, np.array([points], dtype=np.int32), (0, 128, 0))
        # # cv2.imshow('Image', ann_img)
        # # cv2.waitKey(0)
        # # cv2.destroyAllWindows()
        #
        # annotation1 = annotations[32]
        # image_id1 = annotation1['image_id']
        # segmentation1 = annotation1['segmentation']
        # # ann_img = np.zeros((images[image_id1 - 1]['height'], images[image_id1 - 1]['width'], 3)).astype('uint8')
        # points = []
        # for i in range(0, len(segmentation1[0]), 2):
        #     x = segmentation1[0][i]
        #     y = segmentation1[0][i + 1]
        #     point = (x, y)
        #     points.append(point)
        #
        # cv2.fillPoly(ann_img, np.array([points], dtype=np.int32), (0, 128, 0))
        # # cv2.imshow('Image', ann_img)
        # # cv2.waitKey(0)
        # # cv2.destroyAllWindows()
        #
        # annotation1 = annotations[33]
        # image_id1 = annotation1['image_id']
        # segmentation1 = annotation1['segmentation']
        # # ann_img = np.zeros((images[image_id1 - 1]['height'], images[image_id1 - 1]['width'], 3)).astype('uint8')
        # points = []
        # for i in range(0, len(segmentation1[0]), 2):
        #     x = segmentation1[0][i]
        #     y = segmentation1[0][i + 1]
        #     point = (x, y)
        #     points.append(point)
        #
        # cv2.fillPoly(ann_img, np.array([points], dtype=np.int32), (0, 128, 0))
        # cv2.imwrite(images[image_id1 - 1]['file_name'][:-4] + '.png', ann_img)
        # cv2.imshow('Image', ann_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        '''5_1_37_0背景前景标注反了'''
        # annotation1 = annotations[17]
        # image_id1 = annotation1['image_id']
        # segmentation1 = annotation1['segmentation']
        # ann_img = np.zeros((images[image_id1 - 1]['height'], images[image_id1 - 1]['width'], 3)).astype('uint8')
        # points = []
        # for i in range(0, len(segmentation1[0]), 2):
        #     x = segmentation1[0][i]
        #     y = segmentation1[0][i + 1]
        #     point = (x, y)
        #     points.append(point)
        # cv2.fillPoly(ann_img, np.array([points], dtype=np.int32), (0, 0, 128))
        #
        # annotation2 = annotations[16]
        # image_id2 = annotation2['image_id']
        # segmentation2 = annotation2['segmentation']
        # # ann_img = np.zeros((images[image_id2 - 1]['height'], images[image_id2 - 1]['width'], 3)).astype('uint8')
        # points = []
        # for i in range(0, len(segmentation2[0]), 2):
        #     x = segmentation2[0][i]
        #     y = segmentation2[0][i + 1]
        #     point = (x, y)
        #     points.append(point)
        #
        # cv2.fillPoly(ann_img, np.array([points], dtype=np.int32), (0, 128, 0))
        # cv2.imwrite(images[image_id1 - 1]['file_name'][:-4] + '.png', ann_img)
        # cv2.imshow('Image', ann_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()



        '''正常获取mask'''
        # for annotation1, annotation2 in zip(annotations[::2], annotations[1::2]):
        #     image_id1 = annotation1['image_id']
        #     segmentation1 = annotation1['segmentation']
        #     segmentation2 = annotation2['segmentation']
        #
        #     ann_img = np.zeros((images[image_id1 - 1]['height'], images[image_id1 - 1]['width'], 3)).astype('uint8')
        #
        #     points = []
        #     for i in range(0, len(segmentation1[0]), 2):
        #         x = segmentation1[0][i]
        #         y = segmentation1[0][i + 1]
        #         point = (x, y)
        #         points.append(point)
        #
        #     cv2.fillPoly(ann_img, np.array([points], dtype=np.int32), (0, 0, 128))
        #
        #     points = []
        #     for i in range(0, len(segmentation2[0]), 2):
        #         x = segmentation2[0][i]
        #         y = segmentation2[0][i + 1]
        #         point = (x, y)
        #         points.append(point)
        #
        #     cv2.fillPoly(ann_img, np.array([points], dtype=np.int32), (0, 128, 0))




            # # 获取掩膜图像中非零像素的坐标
            # indices = np.where(ann_img != 0)
            # points_inside_region = list(zip(indices[1], indices[0]))

            # cv2.imwrite(images[image_id1 - 1]['file_name'][:-4] + '.png', ann_img)

            # 显示图像
            # cv2.imshow('Image', ann_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()







if __name__ == '__main__':
    main()
    # img = cv2.imread('E:\\ChongQing\\feedport\\Dataset\\Annotations\\1_1_1_1.png')
    #
    # cv2.imshow('Image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
