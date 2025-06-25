import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton,
    QFileDialog, QLabel, QSpinBox, QHBoxLayout, QGraphicsEllipseItem,
    QRubberBand
)
from PyQt6.QtCore import Qt, QPointF, QRect, QSize
from Functions.time_frequency import spectrogram
import pyqtgraph as pg
from matplotlib import cm
from state_annotation.compute import Compute
from pathlib import Path

class RubberbandPlot(pg.PlotWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.origin = None
        self.rubberBand = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        modifiers = event.modifiers()
        if event.button() == Qt.MouseButton.LeftButton and (
            modifiers & Qt.KeyboardModifier.ShiftModifier
        ):
            self.origin = event.pos()
            self.rubberBand.setGeometry(QRect(self.origin, QSize()))
            self.rubberBand.show()
            self.plotItem.vb.setMouseEnabled(x=False, y=False)  # ⛔ disable panning
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.rubberBand.isVisible():
            rect = QRect(self.origin, event.pos()).normalized()
            self.rubberBand.setGeometry(rect)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        modifiers = event.modifiers()
        ctrl_held = modifiers & Qt.KeyboardModifier.ControlModifier
        shift_held = modifiers & Qt.KeyboardModifier.ShiftModifier

        if self.rubberBand.isVisible():
            self.rubberBand.hide()
            rect = self.rubberBand.geometry()

            top_left = self.plotItem.vb.mapSceneToView(self.mapToScene(rect.topLeft()))
            bottom_right = self.plotItem.vb.mapSceneToView(self.mapToScene(rect.bottomRight()))
            xmin, xmax = sorted([top_left.x(), bottom_right.x()])
            ymin, ymax = sorted([top_left.y(), bottom_right.y()])

            for point in self.viewer.state_points:
                pos = point.pos()
                if xmin <= pos.x() <= xmax and ymin <= pos.y() <= ymax:
                    if ctrl_held:
                        point.selected = False
                        point.setBrush(pg.mkBrush('g'))
                    elif shift_held:
                        point.selected = True
                        point.setBrush(pg.mkBrush('y'))

            self.plotItem.vb.setMouseEnabled(x=True, y=True)

class DraggablePoint(QGraphicsEllipseItem):
    def __init__(self, x, y, radius=5, index=None, update_callback=None, selection_callback=None):
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
        self.selection_callback = selection_callback
        self.selected = False

    def toggle_selection(self):
        self.selected = not self.selected
        self.setBrush(pg.mkBrush('y') if self.selected else pg.mkBrush('g'))

    def mousePressEvent(self, event):
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self.toggle_selection()
        else:
            super().mousePressEvent(event)

    def itemChange(self, change, value):
        if change == self.GraphicsItemChange.ItemPositionChange:
            new_y = round(np.clip(value.y(), 0, 21))  # integer range
            new_pos = QPointF(self.fixed_x, new_y)
            if self.update_callback:
                self.update_callback(self.index, new_y)

            # Apply same y to selected points (group move)
            if self.selection_callback and self.selected:
                self.selection_callback(new_y)

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
        self.save_btn = QPushButton('Save Updated State list')
        self.save_btn.clicked.connect(self.save_updated_state_list)
        ctrl.addWidget(self.load_button)
        ctrl.addWidget(self.fs_label)
        ctrl.addWidget(self.fs_input)
        ctrl.addWidget(self.save_btn)
        self.layout.addLayout(ctrl)

        # Plots
        self.plots = [pg.PlotWidget() for _ in range(3)]
        self.plots.append(RubberbandPlot(self))  # last plot with rubberband for easy multiple points selection
        for i, p in enumerate(self.plots):
            p.showGrid(x=True, y=True)
            p.scene().sigMouseMoved.connect(lambda evt, idx=i, plot=p: self.mouse_moved(evt, idx, plot))
            self.layout.addWidget(p)

        for i in range(1,4):
            self.plots[i].setXLink(self.plots[0])

        self.plots[3].setYRange(-0.5, 21.5)

        # variables for state updating
        self.state_y_edit = None # new list of state
        self.state_curve = None  # line to plot the new state
        self.state_points = []   # to check but seems necessary for dragging points  

        #  marker for updating multiple points at once
        self._updating_group = False

        # folder to save updated state file
        self.save_folder = 'data_state_annotation/'

        # variables for changing dispaly of plots
        self.showing_original = True  # flag for plot toggle
        self.toggle_button = QPushButton("Toggle View")
        self.coord_label = QLabel("Cursor: (X, Y)")
        self.layout.addWidget(self.coord_label)
        self.toggle_button.clicked.connect(self.toggle_view)
        self.layout.addWidget(self.toggle_button)

        # labels for plot
        self.labels_power = ['P<sub>δ$', 'P<sub>α', 'P<sub>β', 'P<sub>γ']
        self.labels_power_proportion = ['proportion-P<sub>δ$', 'proportion-P<sub>α', 'proportion-P<sub>β', 'proportion-P<sub>γ']
        self.N_labels = len(self.labels_power)
        self.colors = ['blue', 'red', 'orange', 'green']

        # initialize compute object
        self.C = Compute()

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open .npy file", "recordings_npy/", "NumPy files (*.npy)")
        if not path:
            return
        self.name = path.split('/')[-1]   # get name of recording
        
        #--- check whether or not a save for this recording already exists
        self.D_save = None # initialize to None
        path_save = self.save_folder + 'D_' + self.name
        my_file = Path(path_save)
        if my_file.is_file():
            self.D_save = np.load(path_save, allow_pickle=True).item()

        self.data = np.load(path)
        self.data = self.data - np.median(self.data)
        self.fs = self.fs_input.value()
        t = np.arange(len(self.data)) / self.fs

        #--- send data to compute object
        self.C.get_data(t, self.data, self.fs, 30 * self.fs, 10 *self.fs)
        #--- run to get all variables
        self.C.run()
        #--- call to plot the data 
        self.update_all()

    def update_all(self):
        self.showing_original = True
        self.display_signal()
        self.display_spectrogram()
        self.display_state()
        self.display_editable_state()

    def display_signal(self):
        self.plots[0].clear()
        self.plots[0].setLogMode(y=False)
        t = np.arange(len(self.data)) / self.fs
        self.plots[0].plot(t, self.data, pen='b')
        self.plots[0].setTitle("EEG Signal")
        self.plots[0].setYRange(-75, 75)

    def display_spectrogram(self):
        self.plots[1].clear()
        self.plots[1].setLogMode(y=False)
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
        self.plots[2].plot(self.C.t_list, self.C.state, pen='r', symbol='o')
        self.plots[2].setTitle("State")
        self.plots[2].setYRange(-0.5, 21.5)

    def display_editable_state(self):
        self.plots[3].clear()
        self.state_points = []

        if self.D_save == None:
            self.state_y_edit = self.C.state.copy()
        else:
            self.state_y_edit = self.D_save['state_updated']

        # Plot 4: Editable
        self.state_curve = self.plots[3].plot(self.C.t_list, self.state_y_edit, pen='g', symbol=None)
        self.plots[3].setTitle("Editable State")

        for i in range(len(self.C.t_list)):
            point = DraggablePoint(
                x=self.C.t_list[i],
                y=self.state_y_edit[i],
                index=i,
                update_callback=self.update_point,
                selection_callback=self.group_update_points
            )
            self.plots[3].addItem(point)
            self.state_points.append(point)

    def display_power(self):
        self.plots[0].clear()
        self.plots[0].setLogMode(y=True)
        self.plots[0].addLegend()
        for i in range(self.N_labels):
            self.plots[0].plot(self.C.t_list, self.C.P_signals[i, :], pen=pg.mkPen(self.colors[i], width=2), name = self.labels_power[i])
        self.plots[0].setTitle("Power of the different waves")
        self.plots[1].setYRange(-3, 3)

    def display_power_proportions(self):
        self.plots[1].clear()
        self.plots[1].setLogMode(y=True)
        self.plots[1].addLegend()
        for i in range(self.N_labels):
            self.plots[1].plot(self.C.t_list, self.C.prop_P_signals[i, :], pen=pg.mkPen(self.colors[i], width=2), name = self.labels_power_proportion[i])
        self.plots[1].setTitle("Power proportion of the different waves")
        self.plots[1].setYRange(-5, 0.1)

    def display_supp(self):
        self.plots[2].clear()
        self.plots[2].plot(self.C.t_list, self.C.supp, pen=pg.mkPen(width=2))
        self.plots[2].setTitle("Suppression ratio")
        self.plots[2].setYRange(-0.1, 2.1)

    #-----------------------------------------------------------#
    #---- methods to update the point(s) of the state chart ----#
    #-----------------------------------------------------------#
    def update_point(self, index, new_y):
        self.state_y_edit[index] = new_y
        self.state_curve.setData(self.C.t_list, self.state_y_edit)

    def group_update_points(self, new_y):
        if self._updating_group:
            return  # prevent recursive call

        self._updating_group = True
        try:
            for p in self.state_points:
                if p.selected:
                    p.setPos(QPointF(p.fixed_x, new_y))
                    self.state_y_edit[p.index] = new_y
            self.state_curve.setData(self.C.t_list, self.state_y_edit)
        finally:
            self._updating_group = False

    #-------------------------------------------------------#
    #------ method to visualise current cursor value -------#
    #-------------------------------------------------------#
    def mouse_moved(self, evt, plot_index, plot):
        vb = plot.getViewBox()
        mouse_point = vb.mapSceneToView(evt)
        x = mouse_point.x()
        y = mouse_point.y()
        self.coord_label.setText(f"Plot {plot_index}: Cursor = ({x:.2f}, {y:.2f})")

    #-------------------------------------------------------#
    #-------- method to switch between plot display --------#
    #-------------------------------------------------------#
    def toggle_view(self):
        self.showing_original = not self.showing_original

        # Step 1: Save current x-axis range from the first plot (which is the master for X linking)
        current_x_range = self.plots[0].getViewBox().viewRange()[0]  # [xmin, xmax]

        # Step 2: Update the plots
        if self.showing_original:
            self.display_signal()
            self.display_spectrogram()
            self.display_state()
        else:
            self.display_power()
            self.display_power_proportions()
            self.display_supp()

        # Step 3: Restore x-axis range
        for p in self.plots:
            p.setXRange(*current_x_range, padding=0)
    
    #-------------------------------------------------------#
    #------- save to .npy Dict the edited state list -------#
    #-------------------------------------------------------#
    def save_updated_state_list(self):

        D = {}
        D['t_list'] = self.C.t_list
        D['state'] = self.C.state
        D['state_updated'] = self.state_y_edit
        np.save(self.save_folder + 'D_' + self.name, D)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = EEGViewer()
    viewer.show()
    sys.exit(app.exec())
