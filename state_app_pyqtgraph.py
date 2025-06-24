import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton,
    QFileDialog, QLabel, QSpinBox, QHBoxLayout, QGraphicsEllipseItem
)
from PyQt6.QtCore import Qt, QPointF
from Functions.time_frequency import spectrogram
import pyqtgraph as pg
from matplotlib import cm
from state_annotation.compute import Compute


class DraggablePoint(QGraphicsEllipseItem):
    def __init__(self, x, y, radius=5, index=None, update_callback=None):
        super().__init__(-radius, -radius, 2*radius, 2*radius)
        self.setPos(x, y)
        self.setBrush(pg.mkBrush('g'))
        self.setPen(pg.mkPen('k'))
        self.setFlag(self.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(self.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setFlag(self.GraphicsItemFlag.ItemSendsScenePositionChanges, True)
        self.index = index
        self.fixed_x = x
        self.update_callback = update_callback

    def itemChange(self, change, value):
        if change == self.GraphicsItemChange.ItemPositionChange:
            new_y = round(np.clip(value.y(), 0, 21))  # integer range
            new_pos = QPointF(self.fixed_x, new_y)
            if self.update_callback:
                self.update_callback(self.index, new_y)
            return new_pos
        return super().itemChange(change, value)


class EEGViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Viewer (Draggable Metrics with pyqtgraph)")
        self.resize(1200, 900)

        # UI setup
        self.central = QWidget()
        self.setCentralWidget(self.central)
        self.layout = QVBoxLayout(self.central)

        ctrl = QHBoxLayout()
        self.load_button = QPushButton("Load .npy")
        self.fs_label = QLabel("Sampling freq:")
        self.fs_input = QSpinBox()
        self.fs_input.setRange(1, 10000)
        self.fs_input.setValue(128)
        self.load_button.clicked.connect(self.load_file)
        ctrl.addWidget(self.load_button)
        ctrl.addWidget(self.fs_label)
        ctrl.addWidget(self.fs_input)
        self.layout.addLayout(ctrl)

        # Plots
        self.plots = [pg.PlotWidget() for _ in range(4)]
        for p in self.plots:
            p.showGrid(x=True, y=True)
            self.layout.addWidget(p)

        for i in range(1, 4):
            self.plots[i].setXLink(self.plots[0])

        # Internal state
        self.state_y_edit = None
        self.state_curve = None
        self.state_points = []

        # initialize compute object
        self.C = Compute()

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open .npy file", "", "NumPy files (*.npy)")
        if not path:
            return
        self.data = np.load(path)
        self.fs = self.fs_input.value()
        t = np.arange(len(self.data)) / self.fs

        # send data to compute object
        self.C.get_data(t, self.data, self.fs, 30 * self.fs, 10 *self.fs)
        # run to get all variables
        self.C.run()

        self.update_all()

    def update_all(self):
        self.display_signal()
        self.display_spectrogram()
        self.display_state()

    def display_signal(self):
        self.plots[0].clear()
        t = np.arange(len(self.data)) / self.fs
        self.plots[0].plot(t, self.data, pen='b')
        self.plots[0].setTitle("EEG Signal")

    def display_spectrogram(self):
        self.plots[1].clear()
        t, f, Sxx = spectrogram(self.data, self.fs)
        img = pg.ImageItem(np.log(Sxx.T + 0.0000001))

        # Apply custom colormap
        cmap = cm.get_cmap('rainbow')  # you can try 'plasma', 'inferno', etc.
        lut = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
        img.setLookupTable(lut)
        # fix range for values
        img.setLevels([np.log(0.001), np.log(20)])

        img.setRect(pg.QtCore.QRectF(t[0], f[0], t[-1] - t[0], f[-1] - f[0]))
        self.plots[1].addItem(img)
        self.plots[1].setTitle("Spectrogram")
        self.plots[1].setLabel('left', 'Freq (Hz)')
        self.plots[1].setLabel('bottom', 'Time (s)')

    def display_state(self):
        self.plots[2].clear()
        self.plots[3].clear()
        self.state_points = []

        self.state_y_edit = self.C.state.copy()

        # Plot 3: Static
        self.plots[2].plot(self.C.t_list, self.C.state, pen='r', symbol='o')
        self.plots[2].setTitle("State")

        # Plot 4: Editable
        self.state_curve = self.plots[3].plot(self.C.t_list, self.state_y_edit, pen='g', symbol=None)
        self.plots[3].setTitle("Editable State")

        for i in range(len(self.C.t_list)):
            point = DraggablePoint(
                x=self.C.t_list[i],
                y=self.state_y_edit[i],
                index=i,
                update_callback=self.update_point
            )
            self.plots[3].addItem(point)
            self.state_points.append(point)

    def update_point(self, index, new_y):
        self.state_y_edit[index] = new_y
        self.state_curve.setData(self.C.t_list, self.state_y_edit)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = EEGViewer()
    viewer.show()
    sys.exit(app.exec())
