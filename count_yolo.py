from __future__ import division
import os
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import cv2
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image, letterbox_image
import pandas as pd
import random
import pickle as pkl
import argparse
import math
from timeit import default_timer as timer
from counter import Counter
from distutils.util import strtobool


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


def draw_boxes(out_boxes, out_label_idxs, out_classes, out_colors, img, class_names):
    image = Image.fromarray(img)

    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300

    draw = ImageDraw.Draw(image)
    draw.line((round(image.size[0] * 0.25), 0, round(image.size[0] * 0.25), image.size[1]), fill=(255, 255, 255), width=2)
    draw.line((round(image.size[0] * 0.4), 0, round(image.size[0] * 0.4), image.size[1]), fill=(255, 255, 255), width=2)
    draw.line((round(image.size[0] * 0.6), 0, round(image.size[0] * 0.6), image.size[1]), fill=(255, 255, 255), width=2)
    draw.line((round(image.size[0] * 0.75), 0, round(image.size[0] * 0.75), image.size[1]), fill=(255, 255, 255), width=2)
    del draw

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        label_idx = out_label_idxs[i]
        color = out_colors[i]

        label = '{} {}'.format(predicted_class, label_idx)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        left, top, right, bottom = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        center_x = int((right - left) / 2 + left)
        center_y = int((bottom - top) / 2 + top)

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=color)
        r = 2
        draw.ellipse((center_x - r, center_y - r, center_x + r, center_y + r), fill=color)
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=color)
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

    result = np.asarray(image)

    return result

def arg_parse():
    """
    Parse arguements to the detect module

    """


    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')

    parser.add_argument("--video", dest = 'video', help =
                        "Video to run detection upon",
                        default = "video.avi", type = str)
    parser.add_argument("--dataset", dest = "dataset", help = "Dataset on which the network has been trained", default = "pascal")
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help =
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help =
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help =
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    parser.add_argument("--resize", dest = 'resize', help =
                        "Resize ratio of input video.",
                        default = "0.5", type = str)
    parser.add_argument("--display", dest = 'display', help =
                        "Display result or not.",
                        default = False, type = strtobool)
    parser.add_argument("--update_database", dest = 'update_database', help =
                        "Update database.",
                        default = False, type = strtobool)
    parser.add_argument("--parking_name", dest = 'parking_name', help =
                        "Parking name to update value.",
                        default = None, type = str)
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    resize = float(args.resize)
    display = args.display
    update_database = args.update_database
    parking_name = args.parking_name

    start = 0

    CUDA = torch.cuda.is_available()

    num_classes = 80

    CUDA = torch.cuda.is_available()

    bbox_attrs = 5 + num_classes

    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    if CUDA:
        model.cuda()

    videofile = args.video

    cap = cv2.VideoCapture(videofile)
    video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resize),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize))

    assert cap.isOpened(), 'Cannot capture source'

    frames = 0
    width = 0
    height = 0

    classes = load_classes('data/coco.names')
    counter = Counter(video_size, classes)

    if update_database:
        import MySQLdb
        connection = MySQLdb.connect(
            host='localhost',
            user='root',
            passwd='',
            db='parkingdb')
        cursor = connection.cursor()

        cursor.execute("SELECT enter FROM parking WHERE area = '" + parking_name + "'")
        rows = cursor.fetchall()
        enter_is_right = rows[0][0]
    else:
        connection = None
        cursor = None
        enter_is_right = None

    start = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f'\n[frame {frames:5g}]')

            if not width or not height:
                height, width, _ = frame.shape

            frame = cv2.resize(frame, dsize=(round(width * resize), round(height * resize)))

            start = timer()
            img, orig_im, dim = prep_image(frame, inp_dim)

            im_dim = torch.FloatTensor(dim).repeat(1,2)

            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()

            with torch.no_grad():
                output = model(Variable(img), CUDA)

            output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)
            end = timer()
            print('time:', end - start)

            if type(output) == int:
                frames += 1
                if display:
                    cv2.imshow("frame", orig_im)
                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('q'):
                        break
                continue

            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)

            output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
            output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2

            output[:,1:5] /= scaling_factor

            for i in range(output.shape[0]):
                output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
                output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])

            out_boxes, out_label_idxs, out_classes, out_colors = counter.process_on_frame(orig_im, output, connection, cursor, parking_name, enter_is_right)

            if display:
                result_im = draw_boxes(out_boxes, out_label_idxs, out_classes, out_colors, orig_im, classes)
                cv2.imshow("frame", result_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break

            frames += 1
        else:
            break

    if update_database:
      connection.close()
