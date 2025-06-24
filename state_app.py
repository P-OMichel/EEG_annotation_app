import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton,
    QFileDialog, QLabel, QSpinBox, QHBoxLayout
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

class EEGViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Viewer - Improved Point Dragging")
        self.resize(1200, 1000)

        self.central = QWidget()
        self.setCentralWidget(self.central)
        self.layout = QVBoxLayout(self.central)

        # Controls
        controls = QHBoxLayout()
        self.load_button = QPushButton("Load .npy")
        self.fs_label = QLabel("Sampling freq:")
        self.fs_input = QSpinBox()
        self.fs_input.setRange(1, 5000)
        self.fs_input.setValue(250)
        self.load_button.clicked.connect(self.load_data)
        controls.addWidget(self.load_button)
        controls.addWidget(self.fs_label)
        controls.addWidget(self.fs_input)
        self.layout.addLayout(controls)

        # Matplotlib figure
        self.fig, self.axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)

        # Events
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_drag)
        self.canvas.mpl_connect('button_release_event', self.on_release)

        self.metric_x = None
        self.metric_y = None
        self.selected_idx = None
        self.marker_plot = None

    def load_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open .npy", "", "NumPy Files (*.npy)")
        if not path:
            return
        self.data = np.load(path)
        self.fs = self.fs_input.value()
        self.update_plots()

    def update_plots(self):
        for ax in self.axes:
            ax.clear()

        t = np.arange(len(self.data)) / self.fs

        # EEG Signal
        self.axes[0].plot(t, self.data, color='blue')
        self.axes[0].set_title("EEG Signal")

        # Spectrogram
        f, t_spec, Sxx = spectrogram(self.data, self.fs)
        self.axes[1].pcolormesh(t_spec, f, np.log1p(Sxx), shading='gouraud')
        self.axes[1].set_ylabel("Hz")
        self.axes[1].set_title("Spectrogram")

        # Simulated Metric
        N = 100
        self.metric_x = np.linspace(0, len(self.data)/self.fs, N)
        metric_y_original = np.random.randint(0, 22, N)
        self.metric_y = metric_y_original.copy()

        self.axes[2].plot(self.metric_x, metric_y_original, 'ro-')
        self.axes[2].set_title("Simulated Metric")

        # Editable Plot
        self.marker_plot, = self.axes[3].plot(self.metric_x, self.metric_y, 'go-', picker=5)
        self.axes[3].set_title("Editable Metric (Smooth Dragging)")
        self.fig.tight_layout()
        self.canvas.draw()

    def on_click(self, event):
        if event.inaxes != self.axes[3] or self.metric_x is None:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        dists = np.hypot(self.metric_x - x, self.metric_y - y)
        idx = np.argmin(dists)
        if dists[idx] < 1:  # Selection threshold
            self.selected_idx = idx
            self.marker_plot.set_markerfacecolor('yellow')  # feedback

    def on_drag(self, event):
        if self.selected_idx is None or event.inaxes != self.axes[3]:
            return
        y = round(event.ydata)
        y = np.clip(y, 0, 21)
        self.metric_y[self.selected_idx] = y
        self.marker_plot.set_ydata(self.metric_y)
        self.canvas.draw_idle()

    def on_release(self, event):
        self.selected_idx = None
        self.marker_plot.set_markerfacecolor('green')  # reset color
        self.canvas.draw_idle()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = EEGViewer()
    viewer.show()
    sys.exit(app.exec())

