import sys
import json
import os
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QHBoxLayout, QSlider, QSpinBox, QComboBox, QMessageBox
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
import matplotlib.pyplot as plt

class EEGViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Viewer with Spectrogram")
        self.resize(1000, 1100)

        self.fs = 128
        self.eeg_data = None
        self.start_time = 0
        self.window_size = 30
        self.file_name = None
        self.annotations = {}

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

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
        self.annotation_selector.addItems([str(i) for i in range(21)])
        self.save_btn = QPushButton("Save Annotation")
        self.save_btn.clicked.connect(self.save_annotation)
        self.annotation_layout.addWidget(QLabel("State:"))
        self.annotation_layout.addWidget(self.annotation_selector)
        self.annotation_layout.addWidget(self.save_btn)
        self.layout.addLayout(self.annotation_layout)

        self.fig = Figure(figsize=(10, 10))
        self.canvas = FigureCanvas(self.fig)
        self.ax_timeline = self.fig.add_subplot(511, frame_on=False)
        self.ax_timeline.set_position([0.1, 0.88, 0.8, 0.04])
        self.ax_timeline.set_yticks([])
        self.ax_timeline.set_ylim(0, 1)
        self.ax_timeline.set_title("Annotation Timeline")
        self.ax_timeline.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        self.ax_full = self.fig.add_subplot(512, sharex=self.ax_timeline)
        self.ax_spec = self.fig.add_subplot(513)
        self.ax_spec.sharex(self.ax_full)
        self.ax_segment = self.fig.add_subplot(413)
        self.ax_segment_spec = self.fig.add_subplot(414)
        self.fig.subplots_adjust(hspace=0.3)
        self.layout.addWidget(self.canvas)

        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_drag)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('pick_event', self.on_pick)

        self.start_line = None
        self.end_line = None
        self.start_spec_line = None
        self.end_spec_line = None

        self.interval_start_line = None
        self.interval_end_line = None
        self.interval_mode = False
        self.dragging_interval_line = None

        self.interval_layout = QHBoxLayout()
        self.mark_btn = QPushButton("Mark Interval")
        self.mark_btn.clicked.connect(self.start_interval_selection)

        self.interval_label_selector = QComboBox()
        self.interval_label_selector.addItems(["IES", "Burst", "alpha-supp"])

        self.save_interval_btn = QPushButton("Save Interval")
        self.save_interval_btn.clicked.connect(self.save_interval)

        self.interval_layout.addWidget(self.mark_btn)
        self.interval_layout.addWidget(QLabel("Label:"))
        self.interval_layout.addWidget(self.interval_label_selector)
        self.interval_layout.addWidget(self.save_interval_btn)
        self.layout.addLayout(self.interval_layout)

    def update_fs(self):
        self.fs = self.fs_input.value()

    def load_eeg(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open EEG file", "", "NumPy files (*.npy)")
        if file_path:
            self.file_name = os.path.splitext(os.path.basename(file_path))[0]
            self.eeg_data = np.load(file_path)
            self.start_slider.setMaximum(int(len(self.eeg_data) / self.fs))
            self.window_slider.setMaximum(int(len(self.eeg_data) / self.fs))
            self.window_slider.setValue(30)

            if os.path.exists("annotations.json"):
                with open("annotations.json", "r") as f:
                    self.annotations = json.load(f)
                self.ax_timeline.clear()
                if self.file_name in self.annotations:
                    for state, start_idx, end_idx in self.annotations[self.file_name].get("states", []):
                        t_start = start_idx / self.fs
                        t_end = end_idx / self.fs
                        self.draw_annotation(state, t_start, t_end)
            self.plot_all()

    def update_sliders(self):
        if self.eeg_data is None:
            return
        self.start_time = self.start_slider.value()
        max_time = len(self.eeg_data) / self.fs
        self.window_size = min(self.window_slider.value(), max_time - self.start_time)
        self.plot_all()

    def on_click(self, event):
        if self.interval_mode and event.inaxes == self.ax_segment:
            # Detect if click is near start or end interval line
            x = event.xdata
            if self.interval_start_line and abs(x - self.interval_start_line.get_xdata()[0]) < 0.2:
                self.dragging_interval_line = "start"
            elif self.interval_end_line and abs(x - self.interval_end_line.get_xdata()[0]) < 0.2:
                self.dragging_interval_line = "end"
        elif event.inaxes == self.ax_full:
            # Handle dragging of window boundaries
            if self.start_line and abs(event.xdata - self.start_time) < 0.5:
                self.dragging = 'start'
            elif self.end_line and abs(event.xdata - (self.start_time + self.window_size)) < 0.5:
                self.dragging = 'end'


    def on_drag(self, event):
        if self.eeg_data is None:
            return
        if self.interval_mode and event.inaxes in [self.ax_segment]:
            x = event.xdata
            if self.dragging_interval_line == "start":
                self.interval_start_line.set_xdata([x])
            elif self.dragging_interval_line == "end":
                self.interval_end_line.set_xdata([x])
            self.canvas.draw()
        elif event.inaxes == self.ax_full and hasattr(self, 'dragging') and self.dragging:
            if self.dragging == 'start':
                new_start = max(0, event.xdata)
                if new_start + self.window_size <= len(self.eeg_data) / self.fs:
                    self.start_time = new_start
                    self.start_slider.setValue(int(self.start_time))
            elif self.dragging == 'end':
                new_end = min(len(self.eeg_data) / self.fs, event.xdata)
                if new_end > self.start_time:
                    self.window_size = new_end - self.start_time
                    self.window_slider.setValue(int(self.window_size))
            self.plot_all()

    def on_release(self, event):
        self.dragging = None
        self.dragging_interval_line = None

    def on_pick(self, event):
        artist = event.artist

        # Case 1: Timeline annotation rectangle
        if isinstance(artist, Rectangle) and hasattr(artist, '_annotation_data'):
            state, start_idx, end_idx = artist._annotation_data
            reply = QMessageBox.question(self, "Delete Annotation",
                                        f"Delete annotation {state} at [{start_idx}, {end_idx}]?",
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.annotations[self.file_name]["states"].remove([state, start_idx, end_idx])
                with open("annotations.json", "w") as f:
                    json.dump(self.annotations, f, indent=4)
                artist.remove()
                for text in self.ax_timeline.texts:
                    if text.get_text() == str(state) and abs(text.get_position()[0] - (start_idx + end_idx) / 2 / self.fs) < 0.1:
                        text.remove()
                self.canvas.draw()

        # Case 2: Segment label text
        elif hasattr(artist, '_segment_info'):
            info = artist._segment_info
            reply = QMessageBox.question(self, "Delete Interval Label",
                                        f"Delete interval label '{info['label']}'?",
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                try:
                    self.annotations[self.file_name]["segment label"].remove([
                        info["segment"], info["selection"], info["label"]
                    ])
                    with open("annotations.json", "w") as f:
                        json.dump(self.annotations, f, indent=4)
                    self.plot_all()
                except ValueError:
                    QMessageBox.warning(self, "Error", "Label not found or already deleted.")


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
        self.draw_annotation(state, self.start_time, self.start_time + self.window_size)
        self.canvas.draw()

    def draw_annotation(self, state, t_start, t_end):
        cmap = plt.get_cmap("tab20", 21)
        color = cmap(state)
        color = (*color[:3], 0.3)
        rect = Rectangle((t_start, 0), t_end - t_start, 1, color=color)
        rect.set_picker(True)
        rect._annotation_data = [state, int(t_start * self.fs), int(t_end * self.fs)]
        self.ax_timeline.add_patch(rect)
        self.ax_timeline.text((t_start + t_end) / 2, 0.5 + 0.1 * (state % 3), str(state), ha='center', va='center', fontsize=8, color='black')

    def start_interval_selection(self):
        self.interval_mode = True
        start = self.start_time + self.window_size / 3
        end = self.start_time + 2 * self.window_size / 3
        for ax in [self.ax_segment]:
            self.interval_start_line = ax.axvline(start, color='purple', linestyle='--', linewidth=2)
            self.interval_end_line = ax.axvline(end, color='purple', linestyle='--', linewidth=2)
        self.canvas.draw()

    def save_interval(self):
        if self.eeg_data is None or self.file_name is None:
            return
        if not self.interval_start_line or not self.interval_end_line:
            QMessageBox.warning(self, "No Interval", "Please mark an interval first.")
            return
        selection_start = min(self.interval_start_line.get_xdata()[0], self.interval_end_line.get_xdata()[0])
        selection_end = max(self.interval_start_line.get_xdata()[0], self.interval_end_line.get_xdata()[0])
        label = self.interval_label_selector.currentText()
        segment_start_idx = int(self.start_time * self.fs)
        segment_end_idx = int((self.start_time + self.window_size) * self.fs)
        selection_start_idx = int(selection_start * self.fs)
        selection_end_idx = int(selection_end * self.fs)

        if os.path.exists("annotations.json"):
            with open("annotations.json", "r") as f:
                self.annotations = json.load(f)

        if self.file_name not in self.annotations:
            self.annotations[self.file_name] = {"states": [], "segment label": []}
        elif "segment label" not in self.annotations[self.file_name]:
            self.annotations[self.file_name]["segment label"] = []

        self.annotations[self.file_name]["segment label"].append(
            [[segment_start_idx, segment_end_idx], [selection_start_idx, selection_end_idx], label]
        )

        with open("annotations.json", "w") as f:
            json.dump(self.annotations, f, indent=4)

        QMessageBox.information(self, "Saved", f"Saved interval '{label}'\nSegment: {segment_start_idx}-{segment_end_idx}\nSelection: {selection_start_idx}-{selection_end_idx}")

    def plot_all(self):
        if self.eeg_data is None:
            return
        self.ax_full.clear()
        self.ax_spec.clear()
        self.ax_segment.clear()
        self.ax_segment_spec.clear()

        t = np.arange(len(self.eeg_data)) / self.fs

        self.ax_full.plot(t, self.eeg_data, label="EEG")
        self.ax_full.set_ylim([-75, 75])
        self.ax_full.set_xlim(t[0], t[-1])
        self.start_line = self.ax_full.axvline(self.start_time, color='red', linestyle='--')
        self.end_line = self.ax_full.axvline(self.start_time + self.window_size, color='red', linestyle='--')
        self.ax_full.set_title("Full EEG Signal")
        self.ax_full.set_xlabel("Time (s)")
        self.ax_full.set_ylabel("Amplitude")

        self.ax_spec.specgram(self.eeg_data, Fs=self.fs, cmap='rainbow')
        self.start_spec_line = self.ax_spec.axvline(self.start_time, color='red', linestyle='--')
        self.end_spec_line = self.ax_spec.axvline(self.start_time + self.window_size, color='red', linestyle='--')
        self.ax_spec.set_title("Spectrogram")
        self.ax_spec.set_xlabel("Time (s)")
        self.ax_spec.set_ylabel("Frequency (Hz)")

        start_idx = int(self.start_time * self.fs)
        end_idx = int((self.start_time + self.window_size) * self.fs)
        segment = self.eeg_data[start_idx:end_idx]
        t_segment = np.arange(start_idx, end_idx) / self.fs
        self.ax_segment.plot(t_segment, segment, color='orange')
        self.ax_segment.set_ylim([-75, 75])
        self.ax_segment.set_title(f"Segment: {self.start_time:.2f}s to {self.start_time + self.window_size:.2f}s")
        self.ax_segment.set_xlabel("Time (s)")
        self.ax_segment.set_ylabel("Amplitude")

        if len(segment) > 0:
            self.ax_segment_spec.specgram(segment, Fs=self.fs, cmap='viridis')
            self.ax_segment_spec.set_title("Segment Spectrogram")
            self.ax_segment_spec.set_xlabel("Time (s)")
            self.ax_segment_spec.set_ylabel("Frequency (Hz)")
        
        # === Draw saved segment-level interval labels if they fall inside current segment ===
        if self.file_name in self.annotations and "segment label" in self.annotations[self.file_name]:
            for segment_bounds, selection_bounds, label in self.annotations[self.file_name]["segment label"]:
                seg_start_idx, seg_end_idx = segment_bounds
                if seg_start_idx == start_idx and seg_end_idx == end_idx:
                    sel_start_idx, sel_end_idx = selection_bounds
                    sel_start_time = sel_start_idx / self.fs
                    sel_end_time = sel_end_idx / self.fs

                    # Draw vertical lines in black
                    self.ax_segment.axvline(sel_start_time, color='black', linestyle='--', linewidth=2)
                    self.ax_segment.axvline(sel_end_time, color='black', linestyle='--', linewidth=2)

                    # Draw label text in black and make it clickable
                    t_center = (sel_start_time + sel_end_time) / 2
                    label_text = self.ax_segment.text(
                        t_center, 70, label, color='black', ha='center', va='bottom',
                        fontsize=10, fontweight='bold', picker=20
                    )
                    label_text._segment_info = {
                        "segment": [seg_start_idx, seg_end_idx],
                        "selection": [sel_start_idx, sel_end_idx],
                        "label": label
                    }

        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = EEGViewer()
    viewer.show()
    sys.exit(app.exec())
