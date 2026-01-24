import serial
import numpy as np
from scipy.signal import butter, filtfilt, lfilter, lfilter_zi
import matplotlib.pyplot as plt
from collections import deque

WINDOW_SIZE = 1000
AXIS_UPDATE_INTERVAL = 200

def setup_serial_port():
    ser = None
    try:
        ser = serial.Serial(
            port='/dev/ttyUSB0',
            baudrate=115200,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            timeout=None # Timeout in seconds; set to None for blocking reads
        )
        print(f"Connected to {ser.name}")
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        print("Please check permissions (add user to 'dialout' group) and ensure the port is correct.")
    except KeyboardInterrupt:
        print("Program terminated by user.")

    return ser

def setup_axis():
    plt.ion()
    fig, ax = plt.subplots()
    line_ir, = ax.plot([], [], label="IR")
    
    ax.set_title("Raw PPG Waveform")
    ax.set_xlabel("Samples")
    ax.set_ylabel("ADC Value")
    ax.legend()
    ax.grid()

    return ax, plt, line_ir


def main():
    dc_est = 0.0
    DC_ALPHA = 0.01

    b, a = butter(
        N=3,
        Wn=[0.7, 4],
        btype='band',
        fs=200,
    )

    zi = None
    filter_initialized = False
    sample_count = 0

    ser = setup_serial_port()
    ax, plt, line_ir = setup_axis()

    ymin_hold = None
    ymax_hold = None
    AXIS_MARGIN = 0.15
    
    # raw_buffer = deque(maxlen=WINDOW_SIZE)
    filtered_buffer = deque(maxlen=WINDOW_SIZE)

    while True:
        try:
            # if ser.in_waiting > 0:
            #     line = ser.readline().decode('utf-8').strip()
            line_bytes = ser.readline()
            if not line_bytes:
                continue

            line_str = line_bytes.decode('utf-8').strip()
            ir, red = map(int, line_str.split(','))

            # DC component removal
            dc_est += DC_ALPHA * (ir - dc_est)
            ac = ir - dc_est

            # raw_buffer.append(ir)
    
            if not filter_initialized:
                zi = lfilter_zi(b, a) * ac
                filter_initialized = True

            filtered_sample, zi = lfilter(b, a, [ac], zi=zi)
            filtered_buffer.append(filtered_sample[0])

            sample_count += 1
    
            if len(filtered_buffer) > 100:
                y = np.array(filtered_buffer)
                x = np.arange(len(y))

                y = (y - np.mean(y)) / (np.std(y) + 1e-6)

                line_ir.set_data(x, y)
                
    #             current_axis_min = y.min()
    #             current_axis_max = y.max()
    # 
    #             if ymin_hold is None:
    #                 ymin_hold = current_axis_min
    #                 ymax_hold = current_axis_max
    #             else:
    #                 if current_axis_min < ymin_hold:
    #                     ymin_hold = current_axis_min
    #                 if current_axis_max > ymax_hold:
    #                     ymax_hold = current_axis_max
    # 
    #             span = ymax_hold - ymin_hold + 1e-6
    #             margin = AXIS_MARGIN * span
    
                ax.set_xlim(0, len(y))

                if sample_count % AXIS_UPDATE_INTERVAL == 0:
                    # ax.set_ylim(ymin_hold - margin, ymax_hold + margin)
                    ax.set_ylim(y.min() - 0.5, y.max() + 0.5)
    
                plt.pause(0.001)
        except KeyboardInterrupt:
            print("Stopped by user")
            break
        except Exception as e:
            print(e)

if __name__ == "__main__":
    main()
