from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image, letterbox_image
import pandas as pd
import random
import pickle as pkl
import argparse
import math


write_labels = ['car', 'bus', 'truck']

colors = [(230, 0, 18),
          #(243, 152, 0),
          #(255, 241, 0),
          (143, 195, 31),
          #(0, 153, 68),
          #(0, 158, 150),
          (0, 160, 233),
          #(0, 104, 183),
          #(29, 32, 136),
          (146, 7, 131),
          #(228, 0, 128),
          (229, 0, 79)]


def get_test_input(input_dim, CUDA):
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim))
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    if CUDA:
        img_ = img_.cuda()

    return img_

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

def create_detected_obj_dicts(output, img, previous_detected_obj_dicts):
    detected_obj_dicts = []
    use_label_idxs = []
    '''
    if previous_detected_obj_dicts is None:
        use_label_idxs = []
    else:
        use_label_idxs = [d['label_idx'] for d in previous_detected_obj_dicts]
    '''
    label_idx = 0

    closenesses = [] # 検出した車体間の座標の近さ
    for idx, x in enumerate(output):

        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        left, top, right, bottom = map(int, c1 + c2)
        area = (right - left) * (bottom - top)
        # 小さく映っている車は排除
        if area < 2000:
            continue

        center = (int((right - left) / 2 + left), int((bottom - top) / 2 + top))

        cls = int(x[-1])
        label = "{0}".format(classes[cls])

        if label in write_labels:
            if previous_detected_obj_dicts is None:
                color = colors[label_idx]
                is_first_frame_detected = True

                detected_obj_dict = {'label_idx': label_idx,
                                      'left': left,
                                      'top': top,
                                      'right': right,
                                      'bottom': bottom,
                                      'center': center,
                                      'color': color,
                                      'label': label,
                                      'is_first_frame_detected': is_first_frame_detected}

                # 検出した車体間の座標の近さを計算
                closes = []
                for d in detected_obj_dicts:
                    c_l = abs(left - d['left'])
                    c_t = abs(top - d['top'])
                    c_r = abs(right - d['right'])
                    c_b = abs(bottom - d['bottom'])
                    close = c_l + c_t + c_r + c_b
                    closes.append(close)
                closenesses.append(closes)

                use_label_idxs.append(label_idx)
                detected_obj_dicts.append(detected_obj_dict)
                label_idx += 1
            else:
                # 前フレームで検出された物体と同一の物体か確かめる
                label_idx = None
                min_distance = None
                color = None
                for d in previous_detected_obj_dicts:
                    pre_center = d['center']
                    pre_label_idx = d['label_idx']
                    pre_color = d['color']

                    distance = math.sqrt((center[0] - pre_center[0]) ** 2 + (center[1] - pre_center[1]) ** 2)

                    if distance > 30: # あまりにも離れていたら止める
                        continue
                    if min_distance is None or min_distance > distance:
                        label_idx = pre_label_idx
                        color = pre_color
                        min_distance = distance

                # 前フレームに同一の物体と思われるものがない場合、新しい物体だと認識する
                if min_distance is None or label_idx in use_label_idxs:
                    # 使われていないラベル番号を探す
                    label_idx = 0
                    while True:
                        if not label_idx in use_label_idxs:
                            break
                        label_idx += 1

                    color = colors[label_idx]

                    detected_obj_dict = {'label_idx': label_idx,
                                          'left': left,
                                          'top': top,
                                          'right': right,
                                          'bottom': bottom,
                                          'center': center,
                                          'color': color,
                                          'is_first_frame_detected': True}

                    # 検出した車体間の座標の近さを計算
                    closes = []
                    for d in detected_obj_dicts:
                        c_l = abs(left - d['left'])
                        c_t = abs(top - d['top'])
                        c_r = abs(right - d['right'])
                        c_b = abs(bottom - d['bottom'])
                        close = c_l + c_t + c_r + c_b
                        closes.append(close)
                    closenesses.append(closes)

                    use_label_idxs.append(label_idx)
                    detected_obj_dicts.append(detected_obj_dict)
                else:
                    for d in detected_obj_dicts:
                        if label_idx == d['label_idx']:
                            pre_center = d['center']
                            break

                    speed = center[0] - pre_center[0]
                    detected_obj_dict = {'label_idx': label_idx,
                                          'left': left,
                                          'top': top,
                                          'right': right,
                                          'bottom': bottom,
                                          'center': center,
                                          'color': color,
                                          'is_first_frame_detected': False,
                                          'speed': speed}

                    # 検出した車体間の座標の近さを計算
                    closes = []
                    for d in detected_obj_dicts:
                        c_l = abs(left - d['left'])
                        c_t = abs(top - d['top'])
                        c_r = abs(right - d['right'])
                        c_b = abs(bottom - d['bottom'])
                        close = c_l + c_t + c_r + c_b
                        closes.append(close)
                    closenesses.append(closes)

                    use_label_idxs.append(label_idx)
                    detected_obj_dicts.append(detected_obj_dict)
        use_label_idxs.sort()

    # 同じ物体を複数回検出している問題を解決
    print('closenesses:', closenesses)
    for idx, closes in reversed(list(enumerate(closenesses))):
        for close in closes:
            if close < 50:
                label_idx = detected_obj_dicts[idx]['label_idx']
                del detected_obj_dicts[idx]
                use_label_idxs.remove(label_idx)

    print('use_label_idxs:', use_label_idxs)
    print('detected_obj_dicts:')
    for i, d in enumerate(detected_obj_dicts):
        print(str(i) + ':', d)

    return detected_obj_dicts

def write(detected_obj_dicts, img):
    for d in detected_obj_dicts:
        left = d['left']
        top = d['top']
        right = d['right']
        bottom = d['bottom']
        center = d['center']
        color = d['color']
        label = write_labels[0]

        cv2.rectangle(img, (left, top), (right, bottom), color, 2) # 物体を囲む四角を描画

        cv2.circle(img, center, 3, color, -1) # 物体の中心点を描画

        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        cv2.rectangle(img, (left, top), (left + t_size[0], top + t_size[1]), color, -1) # テキストの背景を描画
        cv2.putText(img, label, (left, top + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1); # テキストを描画
    #return center, left, top, right, bottom
    #return img, center, left, top, right, bottom

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
                        "Input resize ratio of the network. Increase to increase accuracy(?). Decrease to increase speed",
                        default = "1.0", type = str)
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
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

    model(get_test_input(inp_dim, CUDA), CUDA)

    model.eval()

    videofile = args.video

    cap = cv2.VideoCapture(videofile)

    assert cap.isOpened(), 'Cannot capture source'

    frames = 0
    width = 0
    height = 0
    frame_count = 0
    resize_ratio = float(args.resize)
    detected_obj_dicts = None

    start = time.time()
    while cap.isOpened():


        ret, frame = cap.read()
        if ret:
            print()
            print(f'[frame {frame_count:5g}]')

            if not width or not height:
                height, width, _ = frame.shape
                width = round(width * resize_ratio)
                height = round(height * resize_ratio)

            frame = cv2.resize(frame, dsize=(width, height))

            img, orig_im, dim = prep_image(frame, inp_dim)

            im_dim = torch.FloatTensor(dim).repeat(1,2)


            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()

            with torch.no_grad():
                output = model(Variable(img), CUDA)

            output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)

            if type(output) == int:
                frames += 1
                print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
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

            classes = load_classes('data/coco.names')
            #colors = pkl.load(open("pallete", "rb"))

            # 退出判定領域の範囲を描画
            '''
            cv2.rectangle(orig_im, (0, 0), (100, height), (255, 255, 255), -1)
            cv2.rectangle(orig_im, (width - 100, 0), (width, height), (255, 255, 255), -1)
            '''

            previous_detected_obj_dicts = detected_obj_dicts
            detected_obj_dicts = create_detected_obj_dicts(output, orig_im, previous_detected_obj_dicts)
            write(detected_obj_dicts, orig_im)

            cv2.imshow("frame", orig_im)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            # print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))

            frame_count += 1

        else:
            break
