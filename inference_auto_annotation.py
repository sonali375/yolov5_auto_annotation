import torch
import numpy as np
from numpy import random
import cv2
import pybboxes as pbx
from models.experimental import attempt_load
from utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
                                 letterbox, mixup, random_perspective)
# from yolov5processor.utils.datasets import letterbox
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, colorstr,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.general import (check_img_size, non_max_suppression)
# from yolov5processor.utils.torch_utils import select_device

from utils.torch_utils import select_device, smart_inference_mode
from sklearn.metrics import confusion_matrix


class ExecuteInference:
    def __init__(self, weight, confidence=0.1, img_size=320, agnostic_nms=True, gpu=False, iou=0.2):
        self.weight = weight
        self.confidence = confidence
        self.gpu = gpu
        self.iou = iou
        self.agnostic_nms = agnostic_nms
        self.img_size = img_size
        self.device, self.half = self.inference_device()
        self.model, self.names, self.colors = self.load_model()
        print("Loaded Models...")

    def inference_device(self):
        device = select_device('cpu')
        if self.gpu:
            device = select_device(str(torch.cuda.current_device()))
        half = device.type != 'cpu'
        return device, half

    def load_model(self):
        model = attempt_load(self.weight)
        imgsz = check_img_size(self.img_size, s=model.stride.max())
        if self.half:
            model.half()
        names = model.module.names if hasattr(model, 'module') else model.names
        # print("classes: {}".format(names))
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
        img = torch.zeros((1, 3, imgsz, imgsz), device=self.device)
        _ = model(img.half() if self.half else img) if self.device.type != 'cpu' else None
        return model, names, colors

    def predict(self, image):
        image2 = image
        img = letterbox(image, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = self.model(img, augment=False)[0]
        pred = non_max_suppression(pred, self.confidence, self.iou, classes=None, agnostic=self.agnostic_nms)
        _output = list()

        for i, det in enumerate(pred):
            if det is not None and len(det):
                # print(det)
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], image2.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    # print(xyxy)
                    _output.append({"points": xyxy, "conf": conf, "class": cls})
        return _output

import os


def get_scaled_point(iheight, iwidth, points):
    x1, y1, x2, y2 = points
    x1 = x1 / iwidth
    x2 = x2 / iwidth
    y1 = y1 / iheight
    y2 = y2 / iheight

    centerx = (x2 + x1) / 2
    centery = (y2 + y1) / 2

    widthi = y2 - y1
    heighti = x2 - x1
    return centerx, centery, widthi, heighti


yp = ExecuteInference(weight=r"E:\backups\E_backup\Michelin\merged_batch_3 and_for_rej_dok_folder\mixed_both_folders_for_dok_rej\metrices_dok_rej_50_epochs_yolov5x_no_background\weights\best.pt")
directory = r"E:\backups\E_backup\Michelin\blind_test_batch_3\blind_test_for_training\for_dok_rej\annotated_images_dok_rej"
target_dir = r"E:/backups/E_backup/Michelin/blind_test_batch_3/blind_test_for_training/for_dok_rej/annotated/"
# img = cv2.imread(r"C:\Users\sikhin.vc\Documents\loader7_vinod_frames\loader7_vinod_frames\images\Frame_vinod_3768.jpg")
# output = x.predict(img)

# print(output)
# classes = {0: 'LOK', 1: 'SOK', 2: 'VSB', 3: 'PLE', 4: 'NGE', 5: 'LBS', 6: 'POK', 7: 'PNG', 8: '2BAD', 9: 'LCS', 10:'CBS-SOK'}

for each_file in os.listdir(directory):
    title, ext = os.path.splitext(os.path.basename(each_file))
    if ext != ".jpg":
        continue
    # print(each_file)
    image = cv2.imread(os.path.join(directory, each_file))
    image = cv2.resize(image, (320, 320))
    img = image.copy()
    predict = yp.predict(image)
    annotated = False

    for each in predict:
        # if int(each['class']) == 3:
        # cv2.rectangle(img, (int(each['points'][0]), int(each['points'][1])),
        # (int(each['points'][2]), int(each['points'][3])), (255, 0, 0), 2)
        height, width, _ = image.shape
        #
        cv2.imwrite(os.path.join(target_dir, title + ".jpg"), image)

        # cv2.putText(img, classes[int(each['class'])],
        # (int(each['points'][0]), int(each['points'][1])),
        # cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 0, 0), 1)

        with open(os.path.join(target_dir, title + ".txt"), 'a') as f:
            cx, cy, w, h = get_scaled_point(height, width, each['points'])
            f.write("{} {} {} {} {}\n".format(int(each['class']), cx, cy, h, w))

        # annotated = True
        #
        # x1, y1, x2, y2 = each['points']
        # cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        # # cv2.putText(image, "Class 3", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1,
        # # cv2.LINE_AA)
        #
        # elif int(each['class']) == 4:
        cv2.rectangle(img, (int(each['points'][0]), int(each['points'][1])),
                      (int(each['points'][2]), int(each['points'][3])), (0, 0, 255), 2)
        # height, width, _ = image.shape

        # cv2.imwrite(os.path.join(target_dir, title + ".jpg"), image)

        # cv2.putText(img, classes[int(each['class'])],
        # (int(each['points'][0]), int(each['points'][1])),
        # cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 0, 0), 1)

        # with open(os.path.join(target_dir, title + ".txt"), 'w') as f:
        # cx, cy, w, h = get_scaled_point(height, width, each['points'])
        # f.write("{} {} {} {} {} {}\n".format(int(each['class']), each['conf'], cx, cy, h, w))
        #
        # annotated = True
        #
        # x1, y1, x2, y2 = each['points']
        # cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        # # cv2.putText(image, "Class 3", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1,
        # # cv2.LINE_AA)

        # cv2.imwrite(os.path.join(target_dir, title + ".jpg"), image)

        # cv2.imshow("out", img)
        # cv2.waitKey(1)