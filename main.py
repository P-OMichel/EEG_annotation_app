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
import matplotlib.pyplot as plt
from Functions.time_frequency import spectrogram


class EEGViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Viewer with Spectrogram")
        self.resize(1000, 1100)

        # ── data & state ────────────────────────────────────────────────────────
        self.fs = 128
        self.eeg_data = None
        self.start_time = 0
        self.window_size = 30
        self.file_name = None
        self.annotations = {}

        # drag helpers
        self.dragging = None               # 'start' | 'end' | None
        self.interval_mode = False
        self.dragging_interval_line = None

        # ── layout ─────────────────────────────────────────────────────────────
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

        # sliders
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

        # annotation controls
        self.annotation_layout = QHBoxLayout()
        self.annotation_selector = QComboBox()
        self.annotation_selector.addItems([str(i) for i in range(22)])
        self.save_btn = QPushButton("Save Annotation")
        self.save_btn.clicked.connect(self.save_annotation)
        self.annotation_layout.addWidget(QLabel("State:"))
        self.annotation_layout.addWidget(self.annotation_selector)
        self.annotation_layout.addWidget(self.save_btn)
        self.layout.addLayout(self.annotation_layout)

        # matplotlib figure (timeline, full, full-spec, segment, segment-spec)
        self.fig, (self.ax_timeline, self.ax_full,
                   self.ax_spec, self.ax_segment,
                   self.ax_segment_spec) = plt.subplots(
            5, gridspec_kw={'height_ratios': [1, 3, 4, 4, 3]}
        )
        self.fig.set_figheight(10)
        self.fig.subplots_adjust(hspace=0.3)
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)

        # timeline formatting
        self.ax_timeline.set_yticks([])
        self.ax_timeline.set_ylim(0, 1)
        self.ax_timeline.set_title("Annotation Timeline")
        self.ax_timeline.tick_params(axis='x', which='both',
                                     bottom=False, top=False, labelbottom=False)
        self.ax_spec.sharex(self.ax_full)
        self.ax_timeline.sharex(self.ax_full)

        # mpl events
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_drag)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('pick_event', self.on_pick)

        # cursor (vertical lines) refs – filled in by init_plot()
        self.start_line = self.end_line = None
        self.start_spec_line = self.end_spec_line = None

        # interval selection refs
        self.interval_start_line = self.interval_end_line = None

        # interval UI
        self.interval_layout = QHBoxLayout()
        self.mark_btn = QPushButton("Mark Interval")
        self.mark_btn.clicked.connect(self.start_interval_selection)
        self.interval_label_selector = QComboBox()
        self.interval_label_selector.addItems(
            ["IES", "Burst", "alpha-supp",
             "Eye artifacts", "Large artefacts",
             "HF artifacts", "Ground check"]
        )
        self.save_interval_btn = QPushButton("Save Interval")
        self.save_interval_btn.clicked.connect(self.save_interval)
        self.interval_layout.addWidget(self.mark_btn)
        self.interval_layout.addWidget(QLabel("Label:"))
        self.interval_layout.addWidget(self.interval_label_selector)
        self.interval_layout.addWidget(self.save_interval_btn)
        self.layout.addLayout(self.interval_layout)

        self.fig.tight_layout()

    # ────────────────────────────────────────────────────────────────────────
    # basic helpers
    # ────────────────────────────────────────────────────────────────────────
    def update_fs(self):
        self.fs = self.fs_input.value()

    # ────────────────────────────────────────────────────────────────────────
    # loading
    # ────────────────────────────────────────────────────────────────────────
    def load_eeg(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open EEG file", "", "NumPy files (*.npy)")
        if not file_path:
            return

        self.file_name = os.path.splitext(os.path.basename(file_path))[0]
        self.eeg_data = np.load(file_path)

        # block slider signals while we set them up
        self.start_slider.blockSignals(True)
        self.window_slider.blockSignals(True)
        max_seconds = int(len(self.eeg_data) / self.fs)
        self.start_slider.setMaximum(max_seconds)
        self.window_slider.setMaximum(max_seconds)
        self.start_slider.setValue(0)
        self.window_slider.setValue(30)
        self.start_slider.blockSignals(False)
        self.window_slider.blockSignals(False)

        # load annotation file if present
        if os.path.exists("annotations.json"):
            with open("annotations.json", "r") as f:
                self.annotations = json.load(f)
        else:
            self.annotations = {}

        self.init_plot()
        self.canvas.draw_idle()

    # ────────────────────────────────────────────────────────────────────────
    # plotting
    # ────────────────────────────────────────────────────────────────────────
    def init_plot(self):
        """Initial (full) plot – called once after loading a file."""
        self.ax_full.clear()
        self.ax_spec.clear()
        self.ax_timeline.cla()
        self.ax_timeline.set_yticks([])
        self.ax_timeline.set_ylim(0, 1)
        self.ax_timeline.set_title("Annotation Timeline")
        self.ax_timeline.tick_params(
            axis='x', which='both', bottom=False, top=False, labelbottom=False)

        t = np.arange(len(self.eeg_data)) / self.fs
        self.ax_full.plot(t, self.eeg_data, label="EEG")
        self.ax_full.set_ylim([-75, 75])
        self.ax_full.set_xlim(t[0], t[-1])

        # red cursors
        self.start_line = self.ax_full.axvline(
            self.start_time, color='red', linestyle='--')
        self.end_line = self.ax_full.axvline(
            self.start_time + self.window_size, color='red', linestyle='--')

        t_s, f_s, S = spectrogram(self.eeg_data, self.fs)
        self.ax_spec.pcolormesh(
            t_s, f_s, np.log(S + 1e-6),
            shading='nearest', cmap='rainbow',
            vmin=np.log(0.001), vmax=np.log(20))
        self.start_spec_line = self.ax_spec.axvline(
            self.start_time, color='red', linestyle='--')
        self.end_spec_line = self.ax_spec.axvline(
            self.start_time + self.window_size, color='red', linestyle='--')

        # draw existing timeline annotations
        if self.file_name in self.annotations:
            for state, s_idx, e_idx in self.annotations[self.file_name].get("states", []):
                self.draw_annotation(
                    state, s_idx / self.fs, e_idx / self.fs)

        self.update_segment(redraw=False)

    def update_segment(self, redraw=True):
        if self.eeg_data is None:
            return

        self.ax_segment.clear()
        self.ax_segment_spec.clear()

        start_idx = int(self.start_time * self.fs)
        end_idx = int((self.start_time + self.window_size) * self.fs)
        segment = self.eeg_data[start_idx:end_idx]
        t_segment = np.arange(start_idx, end_idx) / self.fs

        self.ax_segment.plot(t_segment, segment, color='orange')
        self.ax_segment.set_ylim([-75, 75])
        self.ax_segment.set_title(
            f"Segment: {self.start_time:.2f}s to {self.start_time + self.window_size:.2f}s")
        self.ax_segment.set_xlabel("Time (s)")
        self.ax_segment.set_ylabel("Amplitude")

        if len(segment) > 0:
            t_spectro_segment, f_spectro_segment, spectro_segment = spectrogram(segment, self.fs)
            self.ax_segment_spec.pcolormesh(
                t_spectro_segment + self.start_time, f_spectro_segment, np.log(spectro_segment + 1e-6),
                shading='nearest', cmap='rainbow', vmin=np.log(0.001), vmax=np.log(20))
            self.ax_segment_spec.set_title("Segment Spectrogram")
            self.ax_segment_spec.set_xlabel("Time (s)")
            self.ax_segment_spec.set_ylabel("Frequency (Hz)")

        # Re-draw interval labels (if any intersect current segment)
        if self.file_name in self.annotations and "segment label" in self.annotations[self.file_name]:
            for segment_bounds, selection_bounds, label in self.annotations[self.file_name]["segment label"]:
                sel_start_idx, sel_end_idx = selection_bounds
                if sel_end_idx > start_idx and sel_start_idx < end_idx:
                    sel_start_time = sel_start_idx / self.fs
                    sel_end_time = sel_end_idx / self.fs
                    self.ax_segment.axvline(sel_start_time, color='black', linestyle='--', linewidth=2)
                    self.ax_segment.axvline(sel_end_time, color='black', linestyle='--', linewidth=2)
                    t_center = (sel_start_time + sel_end_time) / 2
                    label_text = self.ax_segment.text(
                        t_center, 70, label, color='black', ha='center', va='bottom',
                        fontsize=10, fontweight='bold', picker=20
                    )
                    label_text._segment_info = {
                        "segment": segment_bounds,
                        "selection": selection_bounds,
                        "label": label
                    }

        # Re-draw temporary interval lines if marking is active
        if self.interval_mode and self.interval_start_line and self.interval_end_line:
            start_x = self.interval_start_line.get_xdata()[0]
            end_x = self.interval_end_line.get_xdata()[0]
            self.interval_start_line = self.ax_segment.axvline(
                start_x, color='purple', linestyle='--', linewidth=2)
            self.interval_end_line = self.ax_segment.axvline(
                end_x, color='purple', linestyle='--', linewidth=2)

        self.canvas.draw_idle()

    # ────────────────────────────────────────────────────────────────────────
    # slider updates  (still recompute segment)
    # ────────────────────────────────────────────────────────────────────────
    def update_sliders(self):
        if self.eeg_data is None:
            return
        self.start_time = self.start_slider.value()
        max_time = len(self.eeg_data) / self.fs
        self.window_size = min(
            self.window_slider.value(), max_time - self.start_time)

        # move vertical cursors
        self.start_line.set_xdata([self.start_time])
        self.end_line.set_xdata([self.start_time + self.window_size])
        self.start_spec_line.set_xdata([self.start_time])
        self.end_spec_line.set_xdata([self.start_time + self.window_size])

        self.update_segment()

    # ────────────────────────────────────────────────────────────────────────
    # mouse interaction:   click / drag / release
    # ────────────────────────────────────────────────────────────────────────
    def on_click(self, event):
        # interval adjust?
        if self.interval_mode and event.inaxes == self.ax_segment:
            x = event.xdata
            if self.interval_start_line and abs(
                    x - self.interval_start_line.get_xdata()[0]) < 0.2:
                self.dragging_interval_line = "start"
            elif self.interval_end_line and abs(
                    x - self.interval_end_line.get_xdata()[0]) < 0.2:
                self.dragging_interval_line = "end"
            return

        # else: main cursors
        if event.inaxes != self.ax_full or event.xdata is None:
            return
        x = event.xdata
        if abs(x - self.start_line.get_xdata()[0]) < 0.5:
            self.dragging = 'start'
        elif abs(x - self.end_line.get_xdata()[0]) < 0.5:
            self.dragging = 'end'

    def on_drag(self, event):
        if self.eeg_data is None or event.xdata is None:
            return

        # interval dragging
        if self.interval_mode and event.inaxes == self.ax_segment:
            x = event.xdata
            if self.dragging_interval_line == "start":
                self.interval_start_line.set_xdata([x])
            elif self.dragging_interval_line == "end":
                self.interval_end_line.set_xdata([x])
            self.canvas.draw_idle()
            return

        # cursor dragging
        if event.inaxes != self.ax_full or self.dragging is None:
            return

        x = event.xdata
        max_time = len(self.eeg_data) / self.fs

        if self.dragging == 'start':
            new_start = max(0, min(x, self.start_time + self.window_size - 1))
            self.start_time = new_start
            self.start_line.set_xdata([new_start])
            self.start_spec_line.set_xdata([new_start])
            self.start_slider.setValue(int(new_start))

        elif self.dragging == 'end':
            new_end = min(max_time, max(x, self.start_time + 1))
            self.window_size = new_end - self.start_time
            self.end_line.set_xdata([new_end])
            self.end_spec_line.set_xdata([new_end])
            self.window_slider.setValue(int(self.window_size))

        # light refresh only
        self.canvas.draw_idle()

    def on_release(self, event):
        # stop any dragging flags
        self.dragging = None
        self.dragging_interval_line = None
        # heavy refresh (spectrogram etc.)
        self.update_segment()

    # ────────────────────────────────────────────────────────────────────────
    # pick events (delete annotations / labels)
    # ────────────────────────────────────────────────────────────────────────
    def on_pick(self, event):
        artist = event.artist

        # timeline rectangle
        if isinstance(artist, Rectangle) and hasattr(artist, '_annotation_data'):
            state, s_idx, e_idx = artist._annotation_data
            reply = QMessageBox.question(
                self, "Delete Annotation",
                f"Delete annotation {state} at [{s_idx}, {e_idx}]?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return

            self.annotations[self.file_name]["states"].remove(
                [state, s_idx, e_idx])
            with open("annotations.json", "w") as f:
                json.dump(self.annotations, f, indent=4)
            artist.remove()
            # remove matching state text
            for txt in self.ax_timeline.texts:
                if (txt.get_text() == str(state)
                        and abs(txt.get_position()[0] - (s_idx + e_idx) / 2 / self.fs) < 0.1):
                    txt.remove()
            self.canvas.draw_idle()

        # segment label
        elif hasattr(artist, '_segment_info'):
            info = artist._segment_info
            reply = QMessageBox.question(
                self, "Delete Interval Label",
                f"Delete interval label '{info['label']}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return
            try:
                self.annotations[self.file_name]["segment label"].remove(
                    [info["segment"], info["selection"], info["label"]])
                with open("annotations.json", "w") as f:
                    json.dump(self.annotations, f, indent=4)
                self.update_segment()
            except ValueError:
                QMessageBox.warning(self, "Error", "Label not found.")

    # ────────────────────────────────────────────────────────────────────────
    # annotation helpers
    # ────────────────────────────────────────────────────────────────────────
    def save_annotation(self):
        if self.eeg_data is None or self.file_name is None:
            return
        state = int(self.annotation_selector.currentText())
        s_idx = int(self.start_time * self.fs)
        e_idx = int((self.start_time + self.window_size) * self.fs)

        if os.path.exists("annotations.json"):
            with open("annotations.json", "r") as f:
                self.annotations = json.load(f)

        self.annotations.setdefault(
            self.file_name, {"states": [], "segment label": []})

        # overlap check
        for _, es, ee in self.annotations[self.file_name]["states"]:
            if not (e_idx <= es or s_idx >= ee):
                QMessageBox.warning(self, "Overlap Detected",
                                    "This annotation overlaps with an existing one.")
                return

        self.annotations[self.file_name]["states"].append([state, s_idx, e_idx])
        with open("annotations.json", "w") as f:
            json.dump(self.annotations, f, indent=4)

        # draw on timeline
        self.draw_annotation(state, self.start_time,
                             self.start_time + self.window_size)
        self.canvas.draw_idle()

    def draw_annotation(self, state, t_start, t_end):
        cmap = plt.get_cmap("tab20", 21)
        color = (*cmap(state)[:3], 0.3)
        rect = Rectangle((t_start, 0), t_end - t_start, 1, color=color)
        rect.set_picker(True)
        rect._annotation_data = [state, int(t_start * self.fs),
                                 int(t_end * self.fs)]
        self.ax_timeline.add_patch(rect)
        self.ax_timeline.text((t_start + t_end) / 2,
                              0.5 + 0.1 * (state % 3),
                              str(state), ha='center', va='center',
                              fontsize=8, color='black')

    # ────────────────────────────────────────────────────────────────────────
    # interval selection
    # ────────────────────────────────────────────────────────────────────────
    def start_interval_selection(self):
        self.interval_mode = True
        start = self.start_time + self.window_size / 3
        end = self.start_time + 2 * self.window_size / 3
        self.interval_start_line = self.ax_segment.axvline(
            start, color='purple', linestyle='--', linewidth=2)
        self.interval_end_line = self.ax_segment.axvline(
            end, color='purple', linestyle='--', linewidth=2)
        self.canvas.draw_idle()

    def save_interval(self):
        if self.eeg_data is None or self.file_name is None:
            return
        if not (self.interval_start_line and self.interval_end_line):
            QMessageBox.warning(
                self, "No Interval", "Please mark an interval first.")
            return

        sel_start = min(self.interval_start_line.get_xdata()[0],
                        self.interval_end_line.get_xdata()[0])
        sel_end = max(self.interval_start_line.get_xdata()[0],
                      self.interval_end_line.get_xdata()[0])
        label = self.interval_label_selector.currentText()

        seg_s_idx = int(self.start_time * self.fs)
        seg_e_idx = int((self.start_time + self.window_size) * self.fs)
        sel_s_idx = int(sel_start * self.fs)
        sel_e_idx = int(sel_end * self.fs)

        # ensure dict shapes
        if os.path.exists("annotations.json"):
            with open("annotations.json", "r") as f:
                self.annotations = json.load(f)
        self.annotations.setdefault(
            self.file_name, {"states": [], "segment label": []})

        self.annotations[self.file_name]["segment label"].append(
            [[seg_s_idx, seg_e_idx], [sel_s_idx, sel_e_idx], label])

        with open("annotations.json", "w") as f:
            json.dump(self.annotations, f, indent=4)

        QMessageBox.information(
            self, "Saved",
            f"Saved interval '{label}'\n"
            f"Segment: {seg_s_idx}-{seg_e_idx}\n"
            f"Selection: {sel_s_idx}-{sel_e_idx}")
        self.update_segment()

    # ────────────────────────────────────────────────────────────────────────
    # end class
    # ────────────────────────────────────────────────────────────────────────


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = EEGViewer()
    viewer.show()
    sys.exit(app.exec())
