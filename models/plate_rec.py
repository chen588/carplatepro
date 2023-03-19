from models.plateNet import myNet_ocr_color
import torch
import cv2
import numpy as np
import os


def cv_imread(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    return img


def allFilePath(rootPath, allFIleList):
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath, temp)):
            if temp.endswith('.jpg') or temp.endswith('.png') or temp.endswith(
                    '.JPG'):
                allFIleList.append(os.path.join(rootPath, temp))
        else:
            allFilePath(os.path.join(rootPath, temp), allFIleList)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    "cpu")
color = ['黑色', '蓝色', '绿色', '白色', '黄色']
plateName = r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
mean_value, std_value = (0.588, 0.193)


def decodePlate(preds):
    pre = 0
    newPreds = []
    index = []
    for i in range(len(preds)):
        if preds[i] != 0 and preds[i] != pre:
            newPreds.append(preds[i])
            index.append(i)
        pre = preds[i]
    return newPreds, index


def image_processing(img, device):
    img = cv2.resize(img, (168, 48))
    img = np.reshape(img, (48, 168, 3))

    # normalize
    img = img.astype(np.float32)
    img = (img / 255. - mean_value) / std_value
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)

    img = img.to(device)
    img = img.view(1, *img.size())
    return img


def get_plate_result(img, device, model, is_color=False):
    input = image_processing(img, device)
    if is_color:
        preds, color_preds = model(input)
        color_preds = torch.softmax(color_preds, dim=-1)
        color_conf, color_index = torch.max(color_preds, dim=-1)
        color_conf = color_conf.item()
    else:
        preds = model(input)
    preds = torch.softmax(preds, dim=-1)
    prob, index = preds.max(dim=-1)
    index = index.view(-1).detach().cpu().numpy()
    prob = prob.view(-1).detach().cpu().numpy()
    newPreds, new_index = decodePlate(index)
    prob = prob[new_index]
    plate = ""
    for i in newPreds:
        plate += plateName[i]
    if is_color:
        return plate, prob, color[color_index], color_conf
    else:
        return plate, prob


def init_model(device, model_path, is_color=False):
    check_point = torch.load(model_path, map_location=device)
    model_state = check_point['state_dict']
    cfg = check_point['cfg']
    color_classes = 0
    if is_color:
        color_classes = 5
    model = myNet_ocr_color(num_classes=len(plateName),
                            export=True,
                            cfg=cfg,
                            color_num=color_classes)

    model.load_state_dict(model_state, strict=False)
    model.to(device)
    model.eval()
    return model


def get_split_merge(img):
    h, w, c = img.shape
    img_upper = img[0:int(5 / 12 * h), :]
    img_lower = img[int(1 / 3 * h):, :]
    img_upper = cv2.resize(img_upper, (img_lower.shape[1], img_lower.shape[0]))
    new_img = np.hstack((img_upper, img_lower))
    return new_img