import math
import numpy as np

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


class Counter(object):
    def __init__(self, video_size, fps, class_names, resize=0.5):
        print('video_size:', video_size)
        self.class_names = class_names
        self.resize = resize
        self.width, self.height = video_size
        self.fps = fps
        self.used_label_idxs = []
        self.overlap_dicts = []
        self.use_labels = ['car', 'bus', 'truck']
        self.entry_from_left_range = [round(self.width * 0.25), round(self.width * 0.4)]
        self.entry_from_right_range = [round(self.width * 0.6), round(self.width * 0.75)]
        self.exit_to_left_range = [0, round(self.width * 0.25)]
        self.exit_to_right_range = [round(self.width * 0.75), self.width]
        self.entry_from_left_count = 0
        self.entry_from_right_count = 0
        self.exit_to_left_count = 0
        self.exit_to_right_count = 0

        self.pre_detected_obj_dicts = {}

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def process_on_frame(self, image, output, connection, cursor, parking_name, enter_is_right):
        detected_obj_dicts = {}
        use_label_idxs = []

        label_idx = 0

        out_boxes_2 = []
        out_label_idxs = []
        out_scores_2 = []
        out_classes_2 = []
        out_colors_2 = []

        for idx, x in enumerate(output):
            i = idx

            box = [int(y) for y in x[1:5]]
            left, top, right, bottom = box
            cls = int(x[-1])
            predicted_class = self.class_names[cls]
            label = predicted_class

            area = (right - left) * (bottom - top)
            # 小さく映っている車は排除
            if area < 8000 * (self.width * self.height) * 0.000001085069444:
                continue

            center = (int((right - left) / 2 + left), int((bottom - top) / 2 + top))

            # 端っこに映っている車は排除
            if center[0] < self.width * 0.05 or self.width * 0.95 < center[0]:
                continue

            if label in self.use_labels:
                # 前のフレームに車体が無かったとき
                if len(self.pre_detected_obj_dicts) == 0:
                    if idx == 0:
                        self.used_label_idxs = []
                        self.overlap_dicts = []

                    color = colors[label_idx % len(colors)]

                    # 同じ物体を複数回検出している問題を解決
                    break_flag = False
                    for k, v in detected_obj_dicts.items():
                        c_l = abs(left - v['left'])
                        c_t = abs(top - v['top'])
                        c_r = abs(right - v['right'])
                        c_b = abs(bottom - v['bottom'])
                        close = c_l + c_t + c_r + c_b

                        if close < 400 * (self.width * self.height) * 0.000001085069444:
                            break_flag = True
                            break
                    if break_flag: continue

                    # 車体が進入範囲にあるか調べる
                    if self.entry_from_left_range[0] < center[0] < self.entry_from_left_range[1]:
                        is_entried_from_left = True
                        is_entried_from_right = False
                        self.entry_from_left_count += 1
                    elif self.entry_from_right_range[0] < center[0] < self.entry_from_right_range[1]:
                        is_entried_from_left = False
                        is_entried_from_right = True
                        self.entry_from_right_count += 1
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

                    use_label_idxs.append(label_idx)
                    use_label_idxs.sort()
                    if not label_idx in self.used_label_idxs:
                        self.used_label_idxs.append(label_idx)
                    self.used_label_idxs.sort()
                    detected_obj_dicts[label_idx] = detected_obj_dict
                    out_boxes_2.append(box)
                    out_label_idxs.append(label_idx)

                    out_classes_2.append(cls)
                    out_colors_2.append(color)
                    label_idx += 1
                # 前のフレームに車体が有ったとき
                else:
                    # 前フレームで検出された物体と同一の物体か確かめる
                    label_idx = None
                    min_distance = None
                    color = None

                    break_flag = False

                    distance = round(40 * 30 / self.fps)
                    for max_distance in range(distance, distance * 5 + 1, distance): # なるべく一番近いものが優先されるように、ちょっとずつ調べる
                        max_distance *= ((self.width + self.height) / (1280 + 720)) ** 2
                        for k, v in self.pre_detected_obj_dicts.items():
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
                        label_idx = self.used_label_idxs[-1] + 1
                        is_first_frame_detected = True
                        direction = 'unknown'
                        speed = 'unknown'

                        # 車体が進入範囲にあるか調べる
                        if self.entry_from_left_range[0] < center[0] < self.entry_from_left_range[1]:
                            is_entried_from_left = True
                            is_entried_from_right = False
                            self.entry_from_left_count += 1
                        elif self.entry_from_right_range[0] < center[0] < self.entry_from_right_range[1]:
                            is_entried_from_left = False
                            is_entried_from_right = True
                            self.entry_from_right_count += 1
                        else:
                            is_entried_from_left = False
                            is_entried_from_right = False

                        # 車体が退出範囲にあるか調べる
                        is_exited_to_left = False
                        is_exited_to_right = False

                        # 車体が重なりから脱出したものか判定
                        for idx, stack_dict in reversed(list(enumerate(self.overlap_dicts))):
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

                                    self.overlap_dicts.pop(idx)
                            else:
                                self.overlap_dicts.pop(idx)


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

                        use_label_idxs.append(label_idx)
                        use_label_idxs.sort()
                        if not label_idx in self.used_label_idxs:
                            self.used_label_idxs.append(label_idx)
                        self.used_label_idxs.sort()
                        detected_obj_dicts[label_idx] = detected_obj_dict
                        out_boxes_2.append(box)
                        out_label_idxs.append(label_idx)

                        out_classes_2.append(cls)
                        out_colors_2.append(color)
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

                            if close < 400 * (self.width * self.height) * 0.000001085069444:
                                break_flag = True
                                break
                        if break_flag: continue

                        for k, v in detected_obj_dicts.items():
                            if label_idx == k:
                                pre_center = v['center']
                                break

                        # 同フレームにすでに、そのラベル番号が振られた物体がある場合、新しい物体とする
                        if label_idx in use_label_idxs:
                            label_idx = self.used_label_idxs[-1] + 1
                            direction = 'unknown'
                            speed = 'unknown'

                            # 車体が進入範囲にあるか調べる
                            if self.entry_from_left_range[0] < center[0] < self.entry_from_left_range[1]:
                                is_entried_from_left = True
                                is_entried_from_right = False
                                self.entry_from_left_count += 1
                            elif self.entry_from_right_range[0] < center[0] < self.entry_from_right_range[1]:
                                is_entried_from_left = False
                                is_entried_from_right = True
                                self.entry_from_right_count += 1
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
                            is_entried_from_left = self.pre_detected_obj_dicts[label_idx]['is_entried_from_left']
                            is_exited_to_right = self.pre_detected_obj_dicts[label_idx]['is_exited_to_right']
                            is_entried_from_right = self.pre_detected_obj_dicts[label_idx]['is_entried_from_right']
                            is_exited_to_left = self.pre_detected_obj_dicts[label_idx]['is_exited_to_left']
                            if self.entry_from_left_range[0] < center[0] < self.entry_from_left_range[1]:
                                # 左右の進入判定がなく、右への退出判定がない場合
                                if not is_entried_from_left and not is_entried_from_right and not is_exited_to_right:
                                    is_entried_from_left = True
                                    self.entry_from_left_count += 1
                            elif self.entry_from_right_range[0] < center[0] < self.entry_from_right_range[1]:
                                # 左右の進入判定がなく、左への退出判定がない場合
                                if not is_entried_from_left and not is_entried_from_right and not is_exited_to_left:
                                    is_entried_from_right = True
                                    self.entry_from_right_count += 1

                            # 車体が退出範囲にあるか調べる
                            if self.exit_to_left_range[0] < center[0] < self.exit_to_left_range[1]:
                                # 右からの進入判定があり、左への退出判定がない場合
                                if is_entried_from_right and not is_exited_to_left:
                                    is_exited_to_left = True
                                    self.exit_to_left_count += 1

                                    # データベースを更新する
                                    if connection is not None:
                                        if enter_is_right:
                                            cursor.execute("UPDATE parking SET nowcar = nowcar - 1 WHERE area = '" + parking_name + "'")
                                        else:
                                            cursor.execute("UPDATE parking SET nowcar = nowcar + 1 WHERE area = '" + parking_name + "'")
                                        connection.commit()
                            elif self.exit_to_right_range[0] < center[0] < self.exit_to_right_range[1]:
                                # 左からの進入判定があり、右への退出判定がない場合
                                if is_entried_from_left and not is_exited_to_right:
                                    is_exited_to_right = True
                                    self.exit_to_right_count += 1

                                    # データベースを更新する
                                    if connection is not None:
                                        if enter_is_right:
                                            cursor.execute("UPDATE parking SET nowcar = nowcar + 1 WHERE area = '" + parking_name + "'")
                                        else:
                                            cursor.execute("UPDATE parking SET nowcar = nowcar - 1 WHERE area = '" + parking_name + "'")
                                        connection.commit()

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

                        use_label_idxs.append(label_idx)
                        use_label_idxs.sort()
                        if not label_idx in self.used_label_idxs:
                            self.used_label_idxs.append(label_idx)
                        self.used_label_idxs.sort()
                        detected_obj_dicts[label_idx] = detected_obj_dict
                        out_boxes_2.append(box)
                        out_label_idxs.append(label_idx)

                        out_classes_2.append(cls)
                        out_colors_2.append(color)

        vanishing_label_idxs = []
        if len(self.pre_detected_obj_dicts) != 0:
            # 消えたラベル番号を特定
            cur_label_idxs = detected_obj_dicts.keys()
            for k in self.pre_detected_obj_dicts.keys():
                label_idx = k
                if not label_idx in cur_label_idxs:
                    vanishing_label_idxs.append(label_idx)

            # 車体の重なり判定
            pre_label_idxs = self.pre_detected_obj_dicts.keys()
            for vanishing_label_idx in vanishing_label_idxs:
                for label_idx in pre_label_idxs:
                    if vanishing_label_idx != label_idx:
                        # 前フレームで、X座標が重なっているか確認
                        v1 = self.pre_detected_obj_dicts[vanishing_label_idx]
                        v2 = self.pre_detected_obj_dicts[label_idx]

                        l1 = v1['left']
                        r1 = v1['right']
                        l2 = v2['left']
                        r2 = v2['right']

                        if l1 < l2 < r1 < r2 or l2 < l1 < r2 < r1:
                            stack_dict = {}
                            stack_dict[vanishing_label_idx] = self.pre_detected_obj_dicts[vanishing_label_idx]
                            stack_dict[label_idx] = self.pre_detected_obj_dicts[label_idx]
                            self.overlap_dicts.append(stack_dict)

        self.pre_detected_obj_dicts = detected_obj_dicts

        print('use_label_idxs:', use_label_idxs)
        print('vanishing_label_idxs:', vanishing_label_idxs)
        if len(self.overlap_dicts) == 0:
            print('self.overlap_dicts: None')
        else:
            print('self.overlap_dicts:')
            for i, d in enumerate(self.overlap_dicts):
                for k, v in d.items():
                    print('  label_idx:', k, '=>', v)
        print('detected_obj_dicts:')
        for k, v in detected_obj_dicts.items():
            print('  label_idx:', k, '=>', v)

        print('entry_from_left_count:', self.entry_from_left_count)
        print('entry_from_right_count:', self.entry_from_right_count)
        print('exit_to_left_count:', self.exit_to_left_count)
        print('exit_to_right_count:', self.exit_to_right_count)

        # ラベル番号に被りがないかチェック
        label_idxs_for_debug = detected_obj_dicts.keys()

        assert len(label_idxs_for_debug) == len(set(label_idxs_for_debug))

        return out_boxes_2, out_label_idxs, out_classes_2, out_colors_2
