import cv2
import numpy as np
import argparse
import os
import json

from constants import FindRangeConstants
def _parse_args():
    """
    Command-line arguments to the system. 
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='find-rgb-range.py')
    parser.add_argument('--presets', type=str, default=os.path.join(os.getcwd(), 'color-ranges', 'color-values-toponly.json'), help='Presets to load')
   
    args = parser.parse_args()
    return args
def do_nothing(x):
    pass

def set_preset(preset):    
    cv2.setTrackbarPos(f'min_{FindRangeConstants.BLUE}', FindRangeConstants.CV2_TRACKBAR_WINDOW_NAME, preset[0][0])
    cv2.setTrackbarPos(f'min_{FindRangeConstants.GREEN}', FindRangeConstants.CV2_TRACKBAR_WINDOW_NAME, preset[0][1])
    cv2.setTrackbarPos(f'min_{FindRangeConstants.RED}', FindRangeConstants.CV2_TRACKBAR_WINDOW_NAME, preset[0][2])
    cv2.setTrackbarPos(f'max_{FindRangeConstants.BLUE}', FindRangeConstants.CV2_TRACKBAR_WINDOW_NAME, preset[1][0])
    cv2.setTrackbarPos(f'max_{FindRangeConstants.GREEN}', FindRangeConstants.CV2_TRACKBAR_WINDOW_NAME, preset[1][1])
    cv2.setTrackbarPos(f'max_{FindRangeConstants.RED}', FindRangeConstants.CV2_TRACKBAR_WINDOW_NAME, preset[1][2])

def create_slider_window():
    """
    Create adjustment window for masks
    """
    cv2.namedWindow('Track Bars', cv2.WINDOW_NORMAL)
    for color in FindRangeConstants.BGR:
        cv2.createTrackbar(f'min_{color}', FindRangeConstants.CV2_TRACKBAR_WINDOW_NAME, 0, 255, do_nothing)
    for color in FindRangeConstants.BGR:
        cv2.createTrackbar(f'max_{color}', FindRangeConstants.CV2_TRACKBAR_WINDOW_NAME, 0, 255, do_nothing)
def get_colors(path):
    """
    Get colors from json file
    """
    with open(path) as file:
        data = json.load(file)
    colors = {}
    for color in data:
        vals = list(data[color].values())
        colors[color] = (np.array(vals[0:3], dtype=np.uint8), np.array(vals[3:], dtype=np.uint8))
    return colors
def get_all_trackbar_pos():
    min_blue = cv2.getTrackbarPos(f'min_{FindRangeConstants.BLUE}', FindRangeConstants.CV2_TRACKBAR_WINDOW_NAME)
    min_green = cv2.getTrackbarPos(f'min_{FindRangeConstants.GREEN}', FindRangeConstants.CV2_TRACKBAR_WINDOW_NAME)
    min_red = cv2.getTrackbarPos(f'min_{FindRangeConstants.RED}', FindRangeConstants.CV2_TRACKBAR_WINDOW_NAME)
    max_blue = cv2.getTrackbarPos(f'max_{FindRangeConstants.BLUE}', FindRangeConstants.CV2_TRACKBAR_WINDOW_NAME)
    max_green = cv2.getTrackbarPos(f'max_{FindRangeConstants.GREEN}', FindRangeConstants.CV2_TRACKBAR_WINDOW_NAME)
    max_red = cv2.getTrackbarPos(f'max_{FindRangeConstants.RED}', FindRangeConstants.CV2_TRACKBAR_WINDOW_NAME)
    return (min_blue, min_green, min_red), (max_blue, max_green, max_red)
if __name__ == '__main__':
    args = _parse_args()
    FILE_PATH = args.presets
    colors = get_colors(FILE_PATH)
    create_slider_window()

    cap = cv2.VideoCapture(0)
    count = 0
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print("Video Capture FPS:", fps)
    _, frame = cap.read()
    HEIGHT, WIDTH = frame.shape[0], frame.shape[1]
    print("Video Capture Shape:", frame.shape)
    group_frame = np.zeros((frame.shape[0] * 2, frame.shape[1] * 2, 3))
    print("Group Frame Shape", group_frame.shape)
    print("Press the starting character of the Guitar Hero Color you would like to mask out")
    while True: #main loop
        _, frame = cap.read()
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #convertion to HSV
        #reading the trackbar values for thresholds every second
        if count % fps == 0:
            mins, maxs = get_all_trackbar_pos()
        #using inrange function to turn on the image pixels where object threshold is matched
        masked = cv2.inRange(hsv_frame, mins, maxs)
            # cv2.imshow('frame', frame)
            #showing the mask image
            # checking if q key is pressed to break out of loop
            # Using cv2.putText() method
        group_frame[0:HEIGHT, 0:WIDTH] = hsv_frame 
        cv2.putText(group_frame, 'HSV Frame', (WIDTH//2 - WIDTH//8, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 0), thickness=2)
        group_frame[0:HEIGHT, WIDTH:WIDTH*2] = np.stack([masked, masked, masked], axis=-1)
        cv2.putText(group_frame, 'Masked Frame', (WIDTH + WIDTH//2-WIDTH//8, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 0), thickness=2)
        group_frame[HEIGHT:HEIGHT*2, WIDTH:WIDTH*2] = frame
        cv2.putText(group_frame, 'Original Frame', (WIDTH + WIDTH//2-WIDTH//8, HEIGHT + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 0), thickness=2)
        cv2.imshow('Capture', group_frame.astype(np.uint8))
        count += 1
        key = cv2.waitKey(25)
        if key == ord('q'):
            break
        if key == ord('r'):
            set_preset(colors[FindRangeConstants.RED])
        if key == ord('g'):
            set_preset(colors[FindRangeConstants.GREEN])
        if key == ord('b'):
            set_preset(colors[FindRangeConstants.BLUE])
        if key == ord('y'):
            set_preset(colors[FindRangeConstants.YELLOW])
        if key == ord('o'):
            set_preset(colors[FindRangeConstants.ORANGE])
    #printing the threshold values for usage in detection application
    print(f'min_{FindRangeConstants.BLUE} {mins[0]}  min_{FindRangeConstants.GREEN} {mins[1]} min_{FindRangeConstants.RED} {mins[2]}')
    print(f'max_{FindRangeConstants.BLUE} {maxs[0]}  min_{FindRangeConstants.GREEN} {maxs[1]} min_{FindRangeConstants.RED} {maxs[2]}')
    #destroying all windows
    cv2.destroyAllWindows()