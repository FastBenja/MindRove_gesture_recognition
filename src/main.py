import sys
import numpy as np
from scipy.signal import butter, lfilter
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from mindrove.board_shim import BoardShim, BoardIds, MindRoveInputParams


COM_PORT = "COM3"  # Board serial port

params = MindRoveInputParams()
params.serial_port = COM_PORT

board = BoardShim(BoardIds.MINDROVE_WIFI_BOARD, params)
board.prepare_session()
board.start_stream()

board_id = board.get_board_id()
exg_channels = BoardShim.get_exg_channels(board_id)[:8] # Get the first 8 EXG channels
fs = BoardShim.get_sampling_rate(board_id) # Get the sampling rate of the board

# Filter design (Bandpass 10 - 200Hz)
lowcut, highcut = 10.0, 200.0
nyk = 0.5 * fs
b, a = butter(4, [lowcut / nyk, highcut / nyk], btype='band')

zi = np.zeros((max(len(a), len(b)) - 1, 8))  # Initialize filter state (zi) for 8 channels

buffer_size = 1000 # Assuming 500Hz sampling rate, this will give us 2 seconds of data
data_buffer = np.zeros((buffer_size, 8)) # Buffer to hold data for 8 channels

app = QtWidgets.QApplication(sys.argv)
win = pg.GraphicsLayoutWidget(show=True, title="MindRove EEG Data")
win.resize(1000, 600)

curves = []
colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FFA500', '#800080']

for i in range(8):
    p = win.addPlot(row=i, col=0, title=f"Channel {i+1}")
    p.showaxis('left', False)
    p.showaxis('bottom', False)
    p.setYRange(-300, 300)  # Set Y-axis range for EEG data
    curve = p.plot(pen=pg.mkPen(color=colors[i], width=1.5))
    curves.append(curve)

def update():
    global zi, data_buffer

    # Get the latest data from the board
    new_data = board.get_board_data()
    n_new = new_data.shape[1]

    if n_new > 0:
        # Extract the first 8 channels of the new data and format as (samples, channels)
        raw_new = new_data[exg_channels, :].T

        # Filter only new samples
        filt_new, zi = lfilter(b, a, raw_new, axis=0, zi=zi)

        # Shift rolling display buffer
        if n_new >= buffer_size:
            data_buffer = filt_new[-buffer_size:, :].T
        else:
            data_buffer = np.roll(data_buffer, -n_new, axis=1)
            data_buffer[:, -n_new:] = filt_new.T

        for i in range(8):
            curves[i].setData(data_buffer[i, :])


if __name__ == "__main__":
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(50)  # Update every 20 ms

    try:
        sys.exit(app.exec_())
    finally:
        if board.is_prepared():
            board.stop_stream()
            board.release_session()
        print("Application closed and board session released.")