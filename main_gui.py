from gui import Ui_Dialog
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtWidgets
import sys
import argparse
import torch
import copy
import numpy as np
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression_face, scale_coords
from models.plate_rec import get_plate_result, allFilePath, init_model, cv_imread, get_split_merge
from PIL import Image, ImageDraw, ImageFont

clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]


def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  #判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("fonts/platech.ttf",
                                  textSize,
                                  encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def letterbox(img,
              new_shape=(640, 640),
              color=(114, 114, 114),
              auto=True,
              scaleFill=False,
              scaleup=True):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img,
                             top,
                             bottom,
                             left,
                             right,
                             cv2.BORDER_CONSTANT,
                             value=color)
    return img, ratio, (dw, dh)


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = pts.astype('float32')
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0])**2) + ((br[1] - bl[1])**2))
    widthB = np.sqrt(((tr[0] - tl[0])**2) + ((tr[1] - tl[1])**2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0])**2) + ((tr[1] - br[1])**2))
    heightB = np.sqrt(((tl[0] - bl[0])**2) + ((tl[1] - bl[1])**2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]],
                   dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def load_model(weights, device):
    model = attempt_load(weights, map_location=device)
    return model


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0],
                   img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6]] -= pad[0]
    coords[:, [1, 3, 5, 7]] -= pad[1]
    coords[:, :8] /= gain
    coords[:, 0].clamp_(0, img0_shape[1])
    coords[:, 1].clamp_(0, img0_shape[0])
    coords[:, 2].clamp_(0, img0_shape[1])
    coords[:, 3].clamp_(0, img0_shape[0])
    coords[:, 4].clamp_(0, img0_shape[1])
    coords[:, 5].clamp_(0, img0_shape[0])
    coords[:, 6].clamp_(0, img0_shape[1])
    coords[:, 7].clamp_(0, img0_shape[0])

    return coords


def get_plate_rec_landmark(img,
                           xyxy,
                           conf,
                           landmarks,
                           class_num,
                           device,
                           plate_rec_model,
                           is_color=False):
    h, w, c = img.shape
    result_dict = {}
    tl = 1 or round(0.002 * (h + w) / 2) + 1
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    height = y2 - y1
    landmarks_np = np.zeros((4, 2))
    rect = [x1, y1, x2, y2]
    for i in range(4):
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])
        landmarks_np[i] = np.array([point_x, point_y])

    class_label = int(class_num)
    roi_img = four_point_transform(img, landmarks_np)
    if class_label:
        roi_img = get_split_merge(roi_img)
    if not is_color:
        plate_number, rec_prob = get_plate_result(roi_img,
                                                  device,
                                                  plate_rec_model,
                                                  is_color=is_color)
    else:
        plate_number, rec_prob, plate_color, color_conf = get_plate_result(
            roi_img, device, plate_rec_model, is_color=is_color)
    result_dict['rect'] = rect
    result_dict['detect_conf'] = conf
    result_dict['landmarks'] = landmarks_np.tolist()
    result_dict['plate_no'] = plate_number
    result_dict['rec_conf'] = rec_prob
    result_dict['roi_height'] = roi_img.shape[0]
    result_dict['plate_color'] = ""
    if is_color:
        result_dict['plate_color'] = plate_color
        result_dict['color_conf'] = color_conf
    result_dict['plate_type'] = class_label

    return result_dict


def detect_Recognition_plate(model,
                             orgimg,
                             device,
                             plate_rec_model,
                             img_size,
                             is_color=False):
    conf_thres = 0.3
    iou_thres = 0.5
    dict_list = []
    img0 = copy.deepcopy(orgimg)
    assert orgimg is not None, 'Image Not Found '
    h0, w0 = orgimg.shape[:2]
    r = img_size / max(h0, w0)
    if r != 1:
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)),
                          interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())

    img = letterbox(img0, new_shape=imgsz)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    pred = model(img)[0]
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4],
                                      orgimg.shape).round()
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()

            det[:, 5:13] = scale_coords_landmarks(img.shape[2:], det[:, 5:13],
                                                  orgimg.shape).round()

            for j in range(det.size()[0]):
                xyxy = det[j, :4].view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                landmarks = det[j, 5:13].view(-1).tolist()
                class_num = det[j, 13].cpu().numpy()
                result_dict = get_plate_rec_landmark(orgimg,
                                                     xyxy,
                                                     conf,
                                                     landmarks,
                                                     class_num,
                                                     device,
                                                     plate_rec_model,
                                                     is_color=is_color)
                dict_list.append(result_dict)
    return dict_list


def draw_result(orgimg, dict_list):
    result_p = ""
    for result in dict_list:
        rect_area = result['rect']

        x, y, w, h = rect_area[0], rect_area[
            1], rect_area[2] - rect_area[0], rect_area[3] - rect_area[1]
        padding_w = 0.05 * w
        padding_h = 0.11 * h
        rect_area[0] = max(0, int(x - padding_w))
        rect_area[1] = max(0, int(y - padding_h))
        rect_area[2] = min(orgimg.shape[1], int(rect_area[2] + padding_w))
        rect_area[3] = min(orgimg.shape[0], int(rect_area[3] + padding_h))

        height_area = result['roi_height']
        result_p += result['plate_no'] + ' '

        cv2.rectangle(orgimg, (rect_area[0], rect_area[1]),
                      (rect_area[2], rect_area[3]), (0, 0, 255), 2)  #画框
        if len(result) >= 1:
            orgimg = cv2ImgAddText(orgimg, result_p,
                                   rect_area[0] - height_area,
                                   rect_area[1] - height_area - 10,
                                   (0, 255, 0), height_area)

    ui.label_4.setText(result_p)
    ui.show_img(orgimg, ui.graphicsView)
    return orgimg


def get_second(capture):
    if capture.isOpened():
        rate = capture.get(5)
        FrameNumber = capture.get(7)
        duration = FrameNumber / rate
        return int(rate), int(FrameNumber), int(duration)


def pushButton_clicked():
    if ui.file_path is not None:
        if ui.file_path.endswith('.jpg') or ui.file_path.endswith('.png'):
            opt.image_path = ui.file_path
            img = cv_imread(opt.image_path)
            if img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            dict_list = detect_Recognition_plate(detect_model,
                                                 img,
                                                 device,
                                                 plate_rec_model,
                                                 opt.img_size,
                                                 is_color=opt.is_color)
            ui.save_img = draw_result(img, dict_list)
        elif ui.file_path.endswith('.mp4') or ui.file_path.endswith('.avi'):
            img = cv2.cvtColor(ui.detect_img, cv2.COLOR_BGRA2BGR)
            dict_list = detect_Recognition_plate(detect_model,
                                                 img,
                                                 device,
                                                 plate_rec_model,
                                                 opt.img_size,
                                                 is_color=opt.is_color)
            ui.save_img = draw_result(img, dict_list)
            cv2.waitKey(10)

        cv2.destroyAllWindows()
        ui.file_path = None
    else:
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ui.show_img(frame, ui.graphicsView_2)
            img = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            dict_list = detect_Recognition_plate(detect_model,
                                                 img,
                                                 device,
                                                 plate_rec_model,
                                                 opt.img_size,
                                                 is_color=opt.is_color)
            ui.save_img = draw_result(img, dict_list)
            cv2.waitKey(10)


def pushButton_clicked_2():
    sys.exit(app.exec_())


parser = argparse.ArgumentParser()
parser.add_argument('--detect_model',
                    nargs='+',
                    type=str,
                    default='weights/model_detect.pt',
                    help='model.pt path(s)')
parser.add_argument('--rec_model',
                    type=str,
                    default='weights/model_ocr.pth',
                    help='model.pt path(s)')
parser.add_argument('--is_color', type=bool, default=False, help='plate color')
parser.add_argument('--img_size',
                    type=int,
                    default=640,
                    help='inference size (pixels)')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
opt = parser.parse_args()

detect_model = load_model(opt.detect_model, device)
plate_rec_model = init_model(device, opt.rec_model, is_color=opt.is_color)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    widget = QtWidgets.QWidget()
    ui = Ui_Dialog()
    ui.setupUi(widget)
    ui.pushButton_3.clicked.connect(pushButton_clicked)
    ui.pushButton_4.clicked.connect(pushButton_clicked_2)
    widget.show()

    sys.exit(app.exec_())
