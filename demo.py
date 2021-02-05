# -*- coding: utf-8 -*-
'''
@Time          : 20/04/25 15:49
@Author        : huguanghao
@File          : demo.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :
'''

# import sys
# import time
# from PIL import Image, ImageDraw
# from models.tiny_yolo import TinyYoloNet
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import argparse
import copy

"""use CPU or GPU"""
use_cuda = True


def detect_img_folder(cfgfile, weightfile, imgfolder):
    import cv2
    m = Darknet(cfgfile)
    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)

    img_list = os.listdir(imgfolder)
    for imgfile in img_list:
        if imgfile[-3:] == 'jpg' or imgfile[-3:] == 'png':
            img = cv2.imread(imgfolder + imgfile)
            sized = cv2.resize(img, (m.width, m.height))
            sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

            start = time.time()
            boxes = do_detect(m, sized, 0.005, 0.45, use_cuda)
            finish = time.time()

            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))
            # print("bboxes: ", boxes[0])

            """write predicted bboxes into txt file with absolute coordinate form: [cls, conf, x, y, w, h]"""
            cp_boxes = copy.deepcopy(boxes[0])    # [[x1, y1, x2, y2, conf, cls], [], ...]
            img = np.copy(img)
            width = img.shape[1]
            height = img.shape[0]

            # normalised [x1, y1, x2, y2] --> original [x, y, w, h]
            for i in range(len(cp_boxes)):
                cp_boxes[i][0] = (boxes[0][i][0]+boxes[0][i][2])/2 * width
                cp_boxes[i][1] = (boxes[0][i][1]+boxes[0][i][3])/2 * height
                cp_boxes[i][2] = (boxes[0][i][2]-boxes[0][i][0]) * width
                cp_boxes[i][3] = (boxes[0][i][3]-boxes[0][i][1]) * height
            print('【pred boxes】', cp_boxes)

            with open((imgfolder + imgfile[: -4] + '.txt'), 'a+') as a:
                if len(cp_boxes) == 0:
                    a.write("0 0 0 0 0 0")
                else:
                    for j in range(len(cp_boxes)):
                        a.write(str(cp_boxes[j][5]) + ' ' + str(cp_boxes[j][4]) + ' ' \
                              + str(cp_boxes[j][0]) + ' ' + str(cp_boxes[j][1]) + ' ' \
                              + str(cp_boxes[j][2]) + ' ' + str(cp_boxes[j][3]) + '\n')

            plot_boxes_cv2(img, boxes[0], savename='predictions.jpg', class_names=class_names)


def detect_cv2_camera(cfgfile, weightfile):
    import cv2
    m = Darknet(cfgfile)
    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("./test.mp4")
    cap.set(3, 1280)
    cap.set(4, 720)
    print("Starting the YOLO loop...")

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)

    while True:
        ret, img = cap.read()
        sized = cv2.resize(img, (m.width, m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
        finish = time.time()
        print('Predicted in %f seconds.' % (finish - start))

        result_img = plot_boxes_cv2(img, boxes[0], savename=None, class_names=class_names)

        cv2.imshow('Yolo demo', result_img)
        cv2.waitKey(1)

    cap.release()


def detect_skimage(cfgfile, weightfile, imgfile):
    from skimage import io
    from skimage.transform import resize
    m = Darknet(cfgfile)
    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)

    img = io.imread(imgfile)
    sized = resize(img, (m.width, m.height)) * 255

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)


def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-cfgfile',
                        type=str,
                        default='./cfg/yolov4.cfg',
                        help='path of cfg file',
                        dest='cfgfile')
    parser.add_argument('-weightfile',
                        type=str,
                        default='yolov4.weights',
                        help='path of trained model.',
                        dest='weightfile')
    parser.add_argument('-imgfolder',
                        type=str,
                        default='./data/',
                        help='path of your image file.',
                        dest='imgfolder')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    if args.imgfolder:
        detect_img_folder(args.cfgfile, args.weightfile, args.imgfolder)
        # detect_imges(args.cfgfile, args.weightfile)
        # detect_img_folder(args.cfgfile, args.weightfile, args.imgfile)
        # detect_skimage(args.cfgfile, args.weightfile, args.imgfile)
    else:
        detect_cv2_camera(args.cfgfile, args.weightfile)
