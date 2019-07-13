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
ENTRY_RANGES = None
EXIT_RANGES = None
ENTRY_FROM_LEFT_COUNT = 0
ENTRY_FROM_RIGHT_COUNT = 0
EXIT_TO_LEFT_COUNT = 0
EXIT_TO_RIGHT_COUNT = 0

colors = [(18, 0, 230),
          (0, 152, 243),
          (31, 195, 143),
          (68, 153, 0),
          (150, 158, 0),
          (233, 160, 0),
          (183, 104, 0),
          (136, 32, 29),
          (131, 7, 146),
          (128, 0, 228),
          (79, 0, 229)]


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

used_label_idxs = []
stack_dicts = []
def create_detected_obj_dicts(output, img, pre_detected_obj_dicts):
    global used_label_idxs, stack_dicts
    global ENTRY_FROM_LEFT_COUNT, ENTRY_FROM_RIGHT_COUNT, EXIT_TO_LEFT_COUNT, EXIT_TO_RIGHT_COUNT

    height, width, _ = img.shape

    detected_obj_dicts = {}
    use_label_idxs = []

    label_idx = 0

    closenesses = [] # 検出した車体間の座標の近さ
    for idx, x in enumerate(output):

        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        left, top, right, bottom = map(int, c1 + c2)
        area = (right - left) * (bottom - top)
        # 小さく映っている車は排除
        if area < 8000 * resize_ratio ** 2:
            continue

        center = (int((right - left) / 2 + left), int((bottom - top) / 2 + top))

        # 端っこに映っている車は排除
        if center[0] < width * 0.05 or width * 0.95 < center[0]:
            continue

        cls = int(x[-1])
        label = "{0}".format(classes[cls])

        if label in write_labels:
            # 前のフレームに車体が無かったとき
            if len(pre_detected_obj_dicts) == 0:
                if idx == 0:
                    used_label_idxs = []
                    stack_dicts = []

                color = colors[label_idx % len(colors)]

                # 同じ物体を複数回検出している問題を解決
                break_flag = False
                for k, v in detected_obj_dicts.items():
                    c_l = abs(left - v['left'])
                    c_t = abs(top - v['top'])
                    c_r = abs(right - v['right'])
                    c_b = abs(bottom - v['bottom'])
                    close = c_l + c_t + c_r + c_b

                    if close < 400 * resize_ratio ** 2:
                        break_flag = True
                        break
                if break_flag: continue

                # 車体が進入範囲にあるか調べる
                if ENTRY_RANGES[0][0] < center[0] < ENTRY_RANGES[0][1]:
                    is_entried_from_left = True
                    is_entried_from_right = False
                    ENTRY_FROM_LEFT_COUNT += 1
                elif ENTRY_RANGES[1][0] < center[0] < ENTRY_RANGES[1][1]:
                    is_entried_from_left = False
                    is_entried_from_right = True
                    ENTRY_FROM_RIGHT_COUNT += 1
                else:
                    is_entried_from_left = False
                    is_entried_from_right = False

                # 車体が退出範囲にあるか調べる
                is_exited_to_left = False
                is_exited_to_right = False

                detected_obj_dict = {'left': left,
                                     'top': top,
                                     'right': right,
                                     'bottom': bottom,
                                     'center': center,
                                     'color': color,
                                     'is_first_frame_detected': True,
                                     'direction': 'unknown',
                                     'speed': 'unknown',
                                     'is_entried_from_left': is_entried_from_left,
                                     'is_entried_from_right': is_entried_from_right,
                                     'is_exited_to_left': is_exited_to_left,
                                     'is_exited_to_right': is_exited_to_right}

                # 検出した車体間の座標の近さを計算
                closes = []
                for v in detected_obj_dicts.values():
                    c_l = abs(left - v['left'])
                    c_t = abs(top - v['top'])
                    c_r = abs(right - v['right'])
                    c_b = abs(bottom - v['bottom'])
                    close = c_l + c_t + c_r + c_b
                    closes.append(close)
                closenesses.append(closes)

                use_label_idxs.append(label_idx)
                use_label_idxs.sort()
                if not label_idx in used_label_idxs:
                    used_label_idxs.append(label_idx)
                used_label_idxs.sort()
                detected_obj_dicts[label_idx] = detected_obj_dict
                label_idx += 1
            # 前のフレームに車体が有ったとき
            else:
                # 前フレームで検出された物体と同一の物体か確かめる
                label_idx = None
                min_distance = None
                color = None

                break_flag = False
                for max_distance in (40, 80, 120, 160, 200): # なるべく一番近いものが優先されるように、ちょっとずつ調べる
                    max_distance *= resize_ratio ** 2
                    for k, v in pre_detected_obj_dicts.items():
                        pre_center = v['center']
                        pre_label_idx = k
                        pre_color = v['color']

                        # 走行していることを考慮する
                        speed = v['speed']
                        if speed != 'unknown':
                            distance = math.sqrt((center[0] - pre_center[0] - speed) ** 2 + (center[1] - pre_center[1]) ** 2)
                        else:
                            distance = math.sqrt((center[0] - pre_center[0]) ** 2 + (center[1] - pre_center[1]) ** 2)

                        if distance > max_distance: # あまりにも離れていたら止める
                            continue
                        elif min_distance is None or min_distance > distance:
                            label_idx = pre_label_idx
                            color = pre_color
                            min_distance = distance
                            break_flag = True
                            break
                    if break_flag:
                        break

                # 前フレームに同一と思われる物体がない場合（新しい物体だと認識する）
                if min_distance is None:
                    # 最後のラベル番号+1
                    label_idx = used_label_idxs[-1] + 1
                    is_first_frame_detected = True
                    direction = 'unknown'
                    speed = 'unknown'

                    # 車体が進入範囲にあるか調べる
                    if ENTRY_RANGES[0][0] < center[0] < ENTRY_RANGES[0][1]:
                        is_entried_from_left = True
                        is_entried_from_right = False
                        ENTRY_FROM_LEFT_COUNT += 1
                    elif ENTRY_RANGES[1][0] < center[0] < ENTRY_RANGES[1][1]:
                        is_entried_from_left = False
                        is_entried_from_right = True
                        ENTRY_FROM_RIGHT_COUNT += 1
                    else:
                        is_entried_from_left = False
                        is_entried_from_right = False

                    # 車体が退出範囲にあるか調べる
                    is_exited_to_left = False
                    is_exited_to_right = False

                    # 車体が重なりから脱出したものか判定
                    for idx, stack_dict in reversed(list(enumerate(stack_dicts))):
                        stack_label_idx1, stack_label_idx2 = stack_dict.keys()
                        if stack_label_idx2 in detected_obj_dicts:
                            stack_v = detected_obj_dicts[stack_label_idx2]
                            left2 = stack_v['left']
                            right2 = stack_v['right']

                            if left < left2 < right < right2 or left2 < left < right2 < right:
                                label_idx = stack_label_idx1

                                d = stack_dict[label_idx]
                                direction = d['direction']
                                speed = d['speed']
                                is_entried_from_left = d['is_entried_from_left']
                                is_entried_from_right = d['is_entried_from_right']
                                is_exited_to_left = d['is_exited_to_left']
                                is_exited_to_right = d['is_exited_to_right']

                                stack_dicts.pop(idx)
                        else:
                            stack_dicts.pop(idx)


                    color = colors[label_idx % len(colors)]

                    detected_obj_dict = {'left': left,
                                         'top': top,
                                         'right': right,
                                         'bottom': bottom,
                                         'center': center,
                                         'color': color,
                                         'is_first_frame_detected': is_first_frame_detected,
                                         'direction': direction,
                                         'speed': speed,
                                         'is_entried_from_left': is_entried_from_left,
                                         'is_entried_from_right': is_entried_from_right,
                                         'is_exited_to_left': is_exited_to_left,
                                         'is_exited_to_right': is_exited_to_right}

                    # 検出した車体間の座標の近さを計算
                    closes = []
                    for k, v in detected_obj_dicts.items():
                        c_l = abs(left - v['left'])
                        c_t = abs(top - v['top'])
                        c_r = abs(right - v['right'])
                        c_b = abs(bottom - v['bottom'])
                        close = c_l + c_t + c_r + c_b
                        closes.append(close)
                    closenesses.append(closes)

                    use_label_idxs.append(label_idx)
                    use_label_idxs.sort()
                    if not label_idx in used_label_idxs:
                        used_label_idxs.append(label_idx)
                    used_label_idxs.sort()
                    detected_obj_dicts[label_idx] = detected_obj_dict
                # 前フレームに同一と思われる物体がある場合
                else:
                    # 同じ物体を複数回検出している問題を解決
                    break_flag = False
                    for k, v in detected_obj_dicts.items():
                        c_l = abs(left - v['left'])
                        c_t = abs(top - v['top'])
                        c_r = abs(right - v['right'])
                        c_b = abs(bottom - v['bottom'])
                        close = c_l + c_t + c_r + c_b

                        if close < 400 * resize_ratio ** 2:
                            break_flag = True
                            break
                    if break_flag: continue

                    for k, v in detected_obj_dicts.items():
                        if label_idx == k:
                            pre_center = v['center']
                            break

                    # 同フレームにすでに、そのラベル番号が振られた物体がある場合、新しい物体とする
                    if label_idx in use_label_idxs:
                        label_idx = used_label_idxs[-1] + 1
                        direction = 'unknown'
                        speed = 'unknown'

                        # 車体が進入範囲にあるか調べる
                        if ENTRY_RANGES[0][0] < center[0] < ENTRY_RANGES[0][1]:
                            is_entried_from_left = True
                            is_entried_from_right = False
                            ENTRY_FROM_LEFT_COUNT += 1
                        elif ENTRY_RANGES[1][0] < center[0] < ENTRY_RANGES[1][1]:
                            is_entried_from_left = False
                            is_entried_from_right = True
                            ENTRY_FROM_RIGHT_COUNT += 1
                        else:
                            is_entried_from_left = False
                            is_entried_from_right = False

                        # 車体が退出範囲にあるか調べる
                        is_exited_to_left = False
                        is_exited_to_right = False

                        detected_obj_dict = {'left': left,
                                             'top': top,
                                             'right': right,
                                             'bottom': bottom,
                                             'center': center,
                                             'color': color,
                                             'is_first_frame_detected': True,
                                             'direction': 'unknown',
                                             'speed': 'unknown',
                                             'is_entried_from_left': is_entried_from_left,
                                             'is_entried_from_right': is_entried_from_right,
                                             'is_exited_to_left': is_exited_to_left,
                                             'is_exited_to_right': is_exited_to_right}
                    else:
                        speed = center[0] - pre_center[0]
                        direction = 'left' if speed < 0 else 'right'

                        # 車体が進入範囲にあるか調べる
                        is_entried_from_left = pre_detected_obj_dicts[label_idx]['is_entried_from_left']
                        is_exited_to_right = pre_detected_obj_dicts[label_idx]['is_exited_to_right']
                        is_entried_from_right = pre_detected_obj_dicts[label_idx]['is_entried_from_right']
                        is_exited_to_left = pre_detected_obj_dicts[label_idx]['is_exited_to_left']
                        if ENTRY_RANGES[0][0] < center[0] < ENTRY_RANGES[0][1]:
                            # 左右の進入判定がなく、右への退出判定がない場合
                            if not is_entried_from_left and not is_entried_from_right and not is_exited_to_right:
                                is_entried_from_left = True
                                ENTRY_FROM_LEFT_COUNT += 1
                        elif ENTRY_RANGES[1][0] < center[0] < ENTRY_RANGES[1][1]:
                            # 左右の進入判定がなく、左への退出判定がない場合
                            if not is_entried_from_left and not is_entried_from_right and not is_exited_to_left:
                                is_entried_from_right = True
                                ENTRY_FROM_RIGHT_COUNT += 1

                        # 車体が退出範囲にあるか調べる
                        if EXIT_RANGES[0][0] < center[0] < EXIT_RANGES[0][1]:
                            # 右からの進入判定があり、左への退出判定がない場合
                            if is_entried_from_right and not is_exited_to_left:
                                is_exited_to_left = True
                                EXIT_TO_LEFT_COUNT += 1
                        elif EXIT_RANGES[1][0] < center[0] < EXIT_RANGES[1][1]:
                            # 左からの進入判定があり、右への退出判定がない場合
                            if is_entried_from_left and not is_exited_to_right:
                                is_exited_to_right = True
                                EXIT_TO_RIGHT_COUNT += 1

                        detected_obj_dict = {'left': left,
                                             'top': top,
                                             'right': right,
                                             'bottom': bottom,
                                             'center': center,
                                             'color': color,
                                             'is_first_frame_detected': False,
                                             'direction': direction,
                                             'speed': speed,
                                             'is_entried_from_left': is_entried_from_left,
                                             'is_entried_from_right': is_entried_from_right,
                                             'is_exited_to_left': is_exited_to_left,
                                             'is_exited_to_right': is_exited_to_right}

                    # 検出した車体間の座標の近さを計算
                    closes = []
                    for v in detected_obj_dicts.values():
                        c_l = abs(left - v['left'])
                        c_t = abs(top - v['top'])
                        c_r = abs(right - v['right'])
                        c_b = abs(bottom - v['bottom'])
                        close = c_l + c_t + c_r + c_b
                        closes.append(close)
                    closenesses.append(closes)

                    use_label_idxs.append(label_idx)
                    use_label_idxs.sort()
                    if not label_idx in used_label_idxs:
                        used_label_idxs.append(label_idx)
                    used_label_idxs.sort()
                    detected_obj_dicts[label_idx] = detected_obj_dict

    vanishing_label_idxs = []
    if len(pre_detected_obj_dicts) != 0:
        # 消えたラベル番号を特定
        cur_label_idxs = detected_obj_dicts.keys()
        for k in pre_detected_obj_dicts.keys():
            label_idx = k
            if not label_idx in cur_label_idxs:
                vanishing_label_idxs.append(label_idx)

        # 車体の重なり判定
        pre_label_idxs = pre_detected_obj_dicts.keys()
        for vanishing_label_idx in vanishing_label_idxs:
            for label_idx in pre_label_idxs:
                if vanishing_label_idx != label_idx:
                    # 前フレームで、X座標が重なっているか確認
                    v1 = pre_detected_obj_dicts[vanishing_label_idx]
                    v2 = pre_detected_obj_dicts[label_idx]

                    l1 = v1['left']
                    r1 = v1['right']
                    l2 = v2['left']
                    r2 = v2['right']

                    if l1 < l2 < r1 < r2 or l2 < l1 < r2 < r1:
                        stack_dict = {}
                        stack_dict[vanishing_label_idx] = pre_detected_obj_dicts[vanishing_label_idx]
                        stack_dict[label_idx] = pre_detected_obj_dicts[label_idx]
                        stack_dicts.append(stack_dict)

    print('use_label_idxs:', use_label_idxs)
    print('vanishing_label_idxs:', vanishing_label_idxs)
    if len(stack_dicts) == 0:
        print('stack_dicts: None')
    else:
        print('stack_dicts:')
        for i, d in enumerate(stack_dicts):
            print('----------', i, '----------')
            for k, v in d.items():
                print('label_idx:', k, '=>', v)
        print('-----------------------')
    print('detected_obj_dicts:')
    for k, v in detected_obj_dicts.items():
        print('label_idx:', k, '=>', v)

    print('ENTRY_FROM_LEFT_COUNT:', ENTRY_FROM_LEFT_COUNT)
    print('ENTRY_FROM_RIGHT_COUNT:', ENTRY_FROM_RIGHT_COUNT)
    print('EXIT_TO_LEFT_COUNT:', EXIT_TO_LEFT_COUNT)
    print('EXIT_TO_RIGHT_COUNT:', EXIT_TO_RIGHT_COUNT)

    # ラベル番号に被りがないかチェック
    label_idxs_for_debug = detected_obj_dicts.keys()

    assert len(label_idxs_for_debug) == len(set(label_idxs_for_debug))

    return detected_obj_dicts

def write(detected_obj_dicts, img):
    for k, v in detected_obj_dicts.items():
        label_idx = k
        left = v['left']
        top = v['top']
        right = v['right']
        bottom = v['bottom']
        center = v['center']
        color = v['color']
        label = write_labels[0]

        cv2.rectangle(img, (left, top), (right, bottom), color, 2) # 物体を囲む四角を描画

        cv2.circle(img, center, 3, color, -1) # 物体の中心点を描画

        t_size = cv2.getTextSize(str(label_idx), cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        cv2.rectangle(img, (left, top), (left + t_size[0] + 1, top + t_size[1] + 8), color, -1) # テキストの背景を描画
        cv2.putText(img, str(label_idx), (left, top + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1); # テキストを描画
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
    detected_obj_dicts = {}

    start = time.time()
    while cap.isOpened():


        ret, frame = cap.read()
        if ret:
            print(f'\n[frame {frame_count:5g}]')

            '''
            if frame_count < 30:
                frame_count += 1
                continue
            '''

            if not width or not height:
                height, width, _ = frame.shape
                width = round(width * resize_ratio)
                height = round(height * resize_ratio)

                ENTRY_RANGES = [[round(width * 0.25), round(width * 0.4)], [round(width * 0.6), round(width * 0.75)]]
                EXIT_RANGES = [[0, round(width * 0.25)], [round(width * 0.75), width]]

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

            # 侵入・退出判定領域の範囲を描画
            im2 = np.zeros((height, width, 3), np.uint8)
            cv2.rectangle(im2, (ENTRY_RANGES[0][0], 0), (ENTRY_RANGES[0][1], height), (255, 255, 255), -1)
            cv2.rectangle(im2, (ENTRY_RANGES[1][0], 0), (ENTRY_RANGES[1][1], height), (255, 255, 255), -1)
            orig_im = cv2.addWeighted(orig_im, 1, im2, 0.2, 0)


            pre_detected_obj_dicts = detected_obj_dicts
            detected_obj_dicts = create_detected_obj_dicts(output, orig_im, pre_detected_obj_dicts)
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
