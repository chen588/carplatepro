from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
import cv2
from PyQt5.QtGui import *


class Ui_Dialog(QWidget):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(900, 550)
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(300, 370, 111, 41))
        font = QtGui.QFont()
        font.setFamily("Adobe Heiti Std")
        font.setPointSize(12)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.graphicsView = QtWidgets.QGraphicsView(Dialog)
        self.graphicsView.setGeometry(QtCore.QRect(460, 50, 400, 300))
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView_2 = QtWidgets.QGraphicsView(Dialog)
        self.graphicsView_2.setGeometry(QtCore.QRect(45, 50, 400, 300))
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(300, 430, 111, 41))
        font = QtGui.QFont()
        font.setFamily("Adobe Heiti Std")
        font.setPointSize(12)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(Dialog)
        self.pushButton_3.setGeometry(QtCore.QRect(520, 370, 111, 41))
        font = QtGui.QFont()
        font.setFamily("Adobe Heiti Std")
        font.setPointSize(12)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(Dialog)
        self.pushButton_4.setGeometry(QtCore.QRect(520, 430, 111, 41))
        font = QtGui.QFont()
        font.setFamily("Adobe Heiti Std")
        font.setPointSize(12)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setObjectName("pushButton_4")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(210, 20, 71, 31))
        font = QtGui.QFont()
        font.setFamily("Adobe Heiti Std")
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(625, 20, 110, 31))
        font = QtGui.QFont()
        font.setFamily("Adobe Heiti Std")
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(250, 490, 220, 41))
        font = QtGui.QFont()
        font.setFamily("Adobe Heiti Std")
        font.setPointSize(12)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(450, 500, 400, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe Heiti Std")
        font.setPointSize(12)
        self.label_4.setFont(font)
        self.label_4.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_4.setObjectName("label_4")
        font = QtGui.QFont()
        font.setFamily("Adobe Heiti Std")
        font.setPointSize(12)
        self.file_path = None
        self.save_img = None
        self.detect_img = None
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "车牌识别系统"))
        self.pushButton.setText(_translate("Dialog", "选择图片"))
        self.pushButton_2.setText(_translate("Dialog", "保存结果"))
        self.pushButton_3.setText(_translate("Dialog", "开始检测"))
        self.pushButton_4.setText(_translate("Dialog", "退出"))
        self.label.setText(_translate("Dialog", "原始图片"))
        self.label_2.setText(_translate("Dialog", "检测结果"))
        self.label_3.setText(_translate("Dialog", "检测到的车牌是："))
        self.label_4.setText(_translate("Dialog", "None"))
        self.pushButton.clicked.connect(self.choose_file)
        self.pushButton_2.clicked.connect(self.save_file)

    def show_img(self, img, graphicsView):
        graphicsView.show()
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        x = img.shape[1]
        y = img.shape[0]
        if y / 300 >= x / 400:
            ratio = y / 300
        else:
            ratio = x / 400
        img = cv2.resize(img,
                         None,
                         fx=1 / ratio,
                         fy=1 / ratio,
                         interpolation=cv2.INTER_CUBIC)
        x = img.shape[1]
        y = img.shape[0]
        xs = x * 3
        frame = QImage(img, x, y, xs, QImage.Format_BGR888)
        pix = QPixmap.fromImage(frame)
        item = QGraphicsPixmapItem(pix)
        scene = QGraphicsScene()
        scene.addItem(item)
        graphicsView.setScene(scene)

    def choose_file(self):

        self.file_path, _ = QFileDialog.getOpenFileName(
            self, '选择文件', '', 'files(*.*)')
        if self.file_path is not None:
            if self.file_path.endswith('.jpg') or self.file_path.endswith(
                    '.png'):
                self.detect_img = cv2.imread(self.file_path)
                self.show_img(self.detect_img, self.graphicsView_2)

            elif self.file_path.endswith('.mp4') or self.file_path.endswith(
                    '.avi'):
                cap = cv2.VideoCapture(self.file_path)
                while True:
                    ret, self.detect_img = cap.read()
                    if not ret:
                        break
                    self.show_img(self.detect_img, self.graphicsView_2)
                    cv2.waitKey(10)
        

    def save_file(self):
        if self.save_img is not None:
            filesavepath, type = QFileDialog.getSaveFileName(
                self, "文件保存", "/", 'files(*.*)')
            cv2.imwrite(filesavepath, self.save_img)
            self.save_img = None
        else:
            QMessageBox.question(self, '提示！', "请确保图片已经被检测！", QMessageBox.Yes)
