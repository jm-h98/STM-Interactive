import sys
from PyQt5.QtWidgets import (
    QWidget,
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QSlider,
)

from PyQt5.QtGui import QPixmap, QImage, QPalette
from PyQt5.QtCore import Qt


import argparse
import cv2
import numpy as np
import time

from model import model
from utils import (
    load_frames,
    overlay_davis,
)

import qdarkstyle


class App(QWidget):
    def __init__(self):
        super().__init__()

        # buttons
        self.seg_button = QPushButton("Segment This")
        self.seg_button.clicked.connect(self.on_segment)
        self.seg_all_button = QPushButton("Propagate to ALL")
        self.seg_all_button.clicked.connect(self.on_seg_all)
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.on_reset)
        self.done_button = QPushButton("Done")
        self.done_button.clicked.connect(self.on_done)

        # LCD
        self.lcd = QTextEdit()
        self.lcd.setReadOnly(True)
        self.lcd.setMaximumHeight(28)
        self.lcd.setMaximumWidth(100)

        # slide
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)

        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(self.slide)

        self.canvas = QLabel()
        self.canvas.setBackgroundRole(QPalette.Dark)
        self.canvas.setScaledContents(True)

        self.canvas.mousePressEvent = self.on_press
        self.canvas.mouseReleaseEvent = self.on_release
        self.canvas.mouseMoveEvent = self.on_motion

        # navigator
        navi = QHBoxLayout()
        navi.addWidget(self.lcd)
        navi.addStretch(1)
        navi.addWidget(self.seg_button)
        navi.addWidget(self.seg_all_button)
        navi.addWidget(self.reset_button)
        navi.addWidget(self.done_button)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.slider)
        layout.addLayout(navi)
        layout.setStretchFactor(navi, 1)
        layout.setStretchFactor(self.canvas, 0)
        self.setLayout(layout)

        self.re_init()

    def re_init(self):
        self.this_round = -1

        # allow for multiple resolutions
        self.frames = load_frames(
            dir, size=(640, 360), number_of_frames=number_of_frames
        )

        self.num_frames, self.height, self.width = self.frames.shape[:3]
        # # init model
        self.model = model(self.frames)

        # set window
        self.setWindowTitle("Choose the area you want to recolor!")
        self.setGeometry(100, 100, self.width * 2, 2 * self.height)

        self.lcd.setText("{: 3d} / {: 3d}".format(0, self.num_frames - 1))
        self.slider.setMaximum(self.num_frames - 1)

        # initialize action
        self.reset_scribbles()
        self.pressed = False

        # initialize visualize
        self.viz_mode = "fade"
        self.current_mask = np.zeros(
            (self.num_frames, self.height, self.width), dtype=np.uint8
        )
        # self.slider.setValue(10)
        self.cursur = int(self.num_frames / 2.0)
        self.on_showing = None
        self.show_current()

        self.start_time = time.time()
        self.show()

    def toQImage(self, im, copy=False):
        if im is None:
            return QImage()

        if im.dtype == np.uint8:
            if len(im.shape) == 2:
                qim = QImage(
                    im.data,
                    im.shape[1],
                    im.shape[0],
                    im.strides[0],
                    QImage.Format_Indexed8,
                )
                qim.setColorTable(gray_color_table)
                return qim.copy() if copy else qim

            elif len(im.shape) == 3:
                if im.shape[2] == 3:
                    qim = QImage(
                        im.data,
                        im.shape[1],
                        im.shape[0],
                        im.strides[0],
                        QImage.Format_RGB888,
                    )
                    return qim.copy() if copy else qim
                elif im.shape[2] == 4:
                    qim = QImage(
                        im.data,
                        im.shape[1],
                        im.shape[0],
                        im.strides[0],
                        QImage.Format_ARGB32,
                    )
                    return qim.copy() if copy else qim

    def show_current(self):
        viz = overlay_davis(
            self.frames[self.cursur],
            self.current_mask[self.cursur],
            rgb=[0, 0, 128],
        )

        self.current_pixmap = QPixmap.fromImage(self.toQImage(viz))
        self.canvas.setPixmap(self.current_pixmap)

        self.lcd.setText("{: 3d} / {: 3d}".format(self.cursur, self.num_frames - 1))
        self.slider.setValue(self.cursur)
        self.current_viz = viz

    def reset_scribbles(self):
        self.scribbles = {}
        self.scribbles["scribbles"] = [[] for _ in range(self.num_frames)]

    def set_viz_mode(self):
        self.show_current()

    def slide(self):
        self.reset_scribbles()
        self.cursur = self.slider.value()
        self.show_current()
        # print('slide')

    def on_reset(self):
        self.re_init()

    def on_seg_all(self):
        self.model.Segment("All")
        self.current_mask = self.model.Get_mask()
        # clear scribble and reset
        self.show_current()
        self.reset_scribbles()

    def on_done(self):
        pred_masks = self.current_mask

        for i, mask in enumerate(pred_masks):
            cv2.imwrite(
                f"{dir}/{str(i).zfill(5)}_mask_{object_id}.png",
                (mask * 255).astype("uint8"),
            )

    def on_segment(self):
        self.model.Memorize(self.scribbles)
        self.model.Segment(self.cursur)
        self.current_mask[self.cursur] = self.model.Get_mask_index(self.cursur)
        self.show_current()

    def on_press(self, event):
        x = event.pos().x()
        y = event.pos().y()
        norm_x = x / float(self.canvas.size().width())  # event.xdata/self.width
        norm_y = y / float(self.canvas.size().height())  # event.ydata/self.height
        img_x = int(event.pos().x() * self.width / float(self.canvas.size().width()))
        img_y = int(event.pos().y() * self.height / float(self.canvas.size().height()))

        if x and y:
            # print('pressed', x, y)
            self.pressed = True
            self.stroke = {}
            self.stroke["path"] = []
            self.stroke["path"].append([norm_x, norm_y])
            if event.buttons() & Qt.LeftButton:
                self.stroke["object_id"] = 1
            else:
                self.stroke["object_id"] = 0
            self.stroke["start_time"] = time.time()
        self.draw_last_x = img_x
        self.draw_last_y = img_y

    def on_motion(self, event):
        x = event.pos().x()
        y = event.pos().y()
        norm_x = x / float(self.canvas.size().width())  # event.xdata/self.width
        norm_y = y / float(self.canvas.size().height())  # event.ydata/self.height
        img_x = int(event.pos().x() * self.width / float(self.canvas.size().width()))
        img_y = int(event.pos().y() * self.height / float(self.canvas.size().height()))
        if self.pressed and x and y:
            # print('motion', x, y)
            self.stroke["path"].append([norm_x, norm_y])

            if self.stroke["object_id"] == 0:
                cv2.line(
                    self.current_viz,
                    (self.draw_last_x, self.draw_last_y),
                    (img_x, img_y),
                    color=[255, 127, 127],
                    thickness=3,
                )
            else:
                cv2.line(
                    self.current_viz,
                    (self.draw_last_x, self.draw_last_y),
                    (img_x, img_y),
                    color=[127, 255, 127],
                    thickness=3,
                )
            self.canvas.setPixmap(QPixmap.fromImage(self.toQImage(self.current_viz)))

            self.draw_last_x = img_x
            self.draw_last_y = img_y

    def on_release(self, event):
        x = event.pos().x()
        y = event.pos().y()
        norm_x = x / float(self.canvas.size().width())  # event.xdata/self.width
        norm_y = y / float(self.canvas.size().height())  # event.ydata/self.height
        img_x = int(event.pos().x() * self.width / float(self.canvas.size().width()))
        img_y = int(event.pos().y() * self.height / float(self.canvas.size().height()))

        self.pressed = False
        if x and y:
            self.stroke["path"].append([norm_x, norm_y])

        self.stroke["end_time"] = time.time()
        self.scribbles["annotated_frame"] = self.cursur

        self.scribbles["scribbles"][self.cursur].append(self.stroke)


if __name__ == "__main__":

    def get_arguments():
        parser = argparse.ArgumentParser(description="args")
        # add argument for directory
        parser.add_argument("dir", help="Path to the directory containing the frames.")
        parser.add_argument("--nof", type=int, help="Number of frames")
        parser.add_argument("--id", type=int, help="Identifier of the cluster")

        return parser.parse_args()

    args = get_arguments()
    dir = args.dir
    number_of_frames = args.nof
    object_id = args.id

    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    ex = App()
    sys.exit(app.exec_())
