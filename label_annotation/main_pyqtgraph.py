import sys
import json
import os
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QHBoxLayout, QSlider, QSpinBox, QComboBox, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg
from Functions.time_frequency import spectrogram

class EEGViewerPG(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Viewer (PyQtGraph)")
        self.resize(1000, 1000)

        self.fs = 128
        self.eeg_data = None
        self.start_time = 0
        self.window_size = 30
        self.file_name = None
        self.annotations = {}

        self.interval_mode = False
        self.interval_start_line = None
        self.interval_end_line = None
        self.dragging_interval_line = None
        self.timeline_items = []
        self.timeline_texts = []

        self.layout = QVBoxLayout(self)

        self.load_btn = QPushButton("Load EEG (.npy)")
        self.load_btn.clicked.connect(self.load_eeg)
        self.layout.addWidget(self.load_btn)

        self.fs_input = QSpinBox()
        self.fs_input.setValue(self.fs)
        self.fs_input.setRange(1, 10000)
        self.fs_input.valueChanged.connect(self.update_fs)
        self.layout.addWidget(QLabel("Sampling Frequency (Hz):"))
        self.layout.addWidget(self.fs_input)

        self.slider_layout = QHBoxLayout()
        self.start_slider = QSlider(Qt.Orientation.Horizontal)
        self.window_slider = QSlider(Qt.Orientation.Horizontal)
        self.start_slider.valueChanged.connect(self.update_sliders)
        self.window_slider.valueChanged.connect(self.update_sliders)
        self.slider_layout.addWidget(QLabel("Start (s):"))
        self.slider_layout.addWidget(self.start_slider)
        self.slider_layout.addWidget(QLabel("Window (s):"))
        self.slider_layout.addWidget(self.window_slider)
        self.layout.addLayout(self.slider_layout)

        self.annotation_layout = QHBoxLayout()
        self.annotation_selector = QComboBox()
        self.annotation_selector.addItems([str(i) for i in range(22)])
        self.save_btn = QPushButton("Save Annotation")
        self.save_btn.clicked.connect(self.save_annotation)
        self.annotation_layout.addWidget(QLabel("State:"))
        self.annotation_layout.addWidget(self.annotation_selector)
        self.annotation_layout.addWidget(self.save_btn)
        self.layout.addLayout(self.annotation_layout)

        self.interval_layout = QHBoxLayout()
        self.mark_btn = QPushButton("Mark Interval")
        self.mark_btn.clicked.connect(self.start_interval_selection)
        self.interval_label_selector = QComboBox()
        self.interval_label_selector.addItems([
            "IES", "Burst", "alpha-supp", "Eye artifacts",
            "Large artefacts", "HF artifacts", "Ground check"])
        self.save_interval_btn = QPushButton("Save Interval")
        self.save_interval_btn.clicked.connect(self.save_interval)
        self.interval_layout.addWidget(self.mark_btn)
        self.interval_layout.addWidget(QLabel("Label:"))
        self.interval_layout.addWidget(self.interval_label_selector)
        self.interval_layout.addWidget(self.save_interval_btn)
        self.layout.addLayout(self.interval_layout)

        self.plot_widget = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.plot_widget)

        self.timeline_plot = self.plot_widget.addPlot(row=0, col=0, title="Timeline")
        self.full_plot = self.plot_widget.addPlot(row=1, col=0, title="Full EEG")
        self.segment_plot = self.plot_widget.addPlot(row=2, col=0, title="Segment View")
        self.spectrogram_plot = self.plot_widget.addPlot(row=3, col=0, title="Segment Spectrogram")

        self.timeline_plot.hideAxis('left')
        self.timeline_plot.setMouseEnabled(x=False, y=False)

        self.full_curve = self.full_plot.plot(pen='b')
        self.segment_curve = self.segment_plot.plot(pen='r')
        self.image_item = pg.ImageItem()
        self.spectrogram_plot.addItem(self.image_item)

        self.cursor_start = pg.InfiniteLine(angle=90, movable=True, pen='g')
        self.cursor_end = pg.InfiniteLine(angle=90, movable=True, pen='r')
        self.cursor_start.sigPositionChanged.connect(self.update_from_lines)
        self.cursor_end.sigPositionChanged.connect(self.update_from_lines)
        self.full_plot.addItem(self.cursor_start)
        self.full_plot.addItem(self.cursor_end)

        self.timeline_plot.scene().sigMouseClicked.connect(self.handle_pick)

    def update_fs(self):
        self.fs = self.fs_input.value()

    def load_eeg(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open EEG file", "", "NumPy files (*.npy)")
        if file_path:
            self.file_name = os.path.splitext(os.path.basename(file_path))[0]
            self.eeg_data = np.load(file_path)

            duration_sec = int(len(self.eeg_data) / self.fs)
            self.start_slider.setMaximum(duration_sec)
            self.window_slider.setMaximum(duration_sec)
            self.start_slider.setValue(0)
            self.window_slider.setValue(30)

            if os.path.exists("annotations.json"):
                with open("annotations.json", "r") as f:
                    self.annotations = json.load(f)

            t = np.arange(len(self.eeg_data)) / self.fs
            self.full_curve.setData(t, self.eeg_data)
            self.cursor_start.setPos(0)
            self.cursor_end.setPos(30)
            self.update_segment()
            self.draw_timeline()

    def update_sliders(self):
        self.start_time = self.start_slider.value()
        self.window_size = self.window_slider.value()
        self.cursor_start.setPos(self.start_time)
        self.cursor_end.setPos(self.start_time + self.window_size)
        self.update_segment()

    def update_from_lines(self):
        self.start_time = self.cursor_start.value()
        self.window_size = self.cursor_end.value() - self.start_time
        self.start_slider.setValue(int(self.start_time))
        self.window_slider.setValue(int(self.window_size))
        self.update_segment()

    def update_segment(self):
        if self.eeg_data is None:
            return

        start_idx = int(self.start_time * self.fs)
        end_idx = int((self.start_time + self.window_size) * self.fs)
        segment = self.eeg_data[start_idx:end_idx]
        t_segment = np.arange(start_idx, end_idx) / self.fs
        self.segment_curve.setData(t_segment, segment)

        t_s, f_s, Sxx = spectrogram(segment, self.fs)
        Sxx_log = np.log(Sxx + 1e-6)
        self.image_item.setImage(Sxx_log[::-1, :], autoLevels=False)
        self.image_item.scale(t_s[1] - t_s[0], f_s[1] - f_s[0])
        self.image_item.setPos(self.start_time, 0)
        self.spectrogram_plot.setLimits(xMin=self.start_time, xMax=self.start_time + self.window_size)
        self.spectrogram_plot.setYRange(f_s[0], f_s[-1])

        if self.interval_mode and self.interval_start_line and self.interval_end_line:
            self.segment_plot.removeItem(self.interval_start_line)
            self.segment_plot.removeItem(self.interval_end_line)
            start_x = self.interval_start_line.value()
            end_x = self.interval_end_line.value()
            self.interval_start_line = pg.InfiniteLine(pos=start_x, angle=90, movable=True, pen='m')
            self.interval_end_line = pg.InfiniteLine(pos=end_x, angle=90, movable=True, pen='m')
            self.segment_plot.addItem(self.interval_start_line)
            self.segment_plot.addItem(self.interval_end_line)

        if self.file_name in self.annotations and "segment label" in self.annotations[self.file_name]:
            for segment_bounds, selection_bounds, label in self.annotations[self.file_name]["segment label"]:
                sel_start_idx, sel_end_idx = selection_bounds
                if sel_end_idx > start_idx and sel_start_idx < end_idx:
                    sel_start_time = sel_start_idx / self.fs
                    sel_end_time = sel_end_idx / self.fs
                    self.segment_plot.addItem(pg.InfiniteLine(sel_start_time, angle=90, pen='k'))
                    self.segment_plot.addItem(pg.InfiniteLine(sel_end_time, angle=90, pen='k'))
                    t_center = (sel_start_time + sel_end_time) / 2
                    text = pg.TextItem(text=label, color='black', anchor=(0.5, 1.0))
                    text.setPos(t_center, 70)
                    self.segment_plot.addItem(text)

    def draw_timeline(self):
        self.timeline_plot.clear()
        self.timeline_items.clear()
        self.timeline_texts.clear()

        if self.file_name not in self.annotations:
            return
        for state, s_idx, e_idx in self.annotations[self.file_name].get("states", []):
            t_start = s_idx / self.fs
            t_end = e_idx / self.fs
            rect = pg.LinearRegionItem(values=[t_start, t_end], movable=False)
            rect.setZValue(-10)
            rect._annotation_data = (state, s_idx, e_idx)
            self.timeline_plot.addItem(rect)
            self.timeline_items.append(rect)
            label = pg.TextItem(text=str(state), anchor=(0.5, 0))
            label.setPos((t_start + t_end) / 2, 0.5)
            self.timeline_plot.addItem(label)
            self.timeline_texts.append(label)

    def handle_pick(self, event):
        items = self.timeline_plot.items()
        pos = event.scenePos()
        for item in items:
            if hasattr(item, '_annotation_data') and item.sceneBoundingRect().contains(pos):
                state, s_idx, e_idx = item._annotation_data
                reply = QMessageBox.question(self, "Delete Annotation",
                                             f"Delete annotation {state} at [{s_idx}, {e_idx}]?",
                                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.Yes:
                    self.annotations[self.file_name]["states"].remove([state, s_idx, e_idx])
                    with open("annotations.json", "w") as f:
                        json.dump(self.annotations, f, indent=4)
                    self.draw_timeline()
                    break

    def save_annotation(self):
        if self.eeg_data is None or self.file_name is None:
            return

        state = int(self.annotation_selector.currentText())
        start_idx = int(self.start_time * self.fs)
        end_idx = int((self.start_time + self.window_size) * self.fs)

        if os.path.exists("annotations.json"):
            with open("annotations.json", "r") as f:
                self.annotations = json.load(f)

        if self.file_name not in self.annotations:
            self.annotations[self.file_name] = {"states": [], "segment label": []}

        for _, existing_start, existing_end in self.annotations[self.file_name]["states"]:
            if not (end_idx <= existing_start or start_idx >= existing_end):
                QMessageBox.warning(self, "Overlap Detected", "This annotation overlaps with an existing one and will not be saved.")
                return

        self.annotations[self.file_name]["states"].append([state, start_idx, end_idx])
        with open("annotations.json", "w") as f:
            json.dump(self.annotations, f, indent=4)
        self.draw_timeline()
        self.update_segment()

    def start_interval_selection(self):
        self.interval_mode = True
        start = self.start_time + self.window_size / 3
        end = self.start_time + 2 * self.window_size / 3
        self.interval_start_line = pg.InfiniteLine(pos=start, angle=90, movable=True, pen='m')
        self.interval_end_line = pg.InfiniteLine(pos=end, angle=90, movable=True, pen='m')
        self.segment_plot.addItem(self.interval_start_line)
        self.segment_plot.addItem(self.interval_end_line)

    def save_interval(self):
        if self.eeg_data is None or self.file_name is None:
            return
        if not self.interval_start_line or not self.interval_end_line:
            QMessageBox.warning(self, "No Interval", "Please mark an interval first.")
            return

        sel_start = min(self.interval_start_line.value(), self.interval_end_line.value())
        sel_end = max(self.interval_start_line.value(), self.interval_end_line.value())
        label = self.interval_label_selector.currentText()

        seg_s_idx = int(self.start_time * self.fs)
        seg_e_idx = int((self.start_time + self.window_size) * self.fs)
        sel_s_idx = int(sel_start * self.fs)
        sel_e_idx = int(sel_end * self.fs)

        if os.path.exists("annotations.json"):
            with open("annotations.json", "r") as f:
                self.annotations = json.load(f)

        self.annotations.setdefault(self.file_name, {"states": [], "segment label": []})
        self.annotations[self.file_name]["segment label"].append(
            [[seg_s_idx, seg_e_idx], [sel_s_idx, sel_e_idx], label])

        with open("annotations.json", "w") as f:
            json.dump(self.annotations, f, indent=4)

        QMessageBox.information(self, "Saved", f"Saved interval '{label}'\nSegment: {seg_s_idx}-{seg_e_idx}\nSelection: {sel_s_idx}-{sel_e_idx}")
        self.update_segment()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = EEGViewerPG()
    viewer.show()
    sys.exit(app.exec())
