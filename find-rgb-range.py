import cv2
import numpy as np
import argparse
import os
import json
from utils import find_color
from constants import FindRangeConstants

def _parse_args():
    """ Parse command line arguments for find-rgb-range.py
    Returns:
        namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='find-rgb-range.py')
    parser.add_argument('--presets', type=str, default=os.path.join(os.getcwd(), 'color-ranges', 'color-values-toponly.json'), help='Presets to load')
    parser.add_argument('--area_threshold', type=int, default=1000, help="Threshold at which we consider a contour big enough to recognize")
    args = parser.parse_args()
    return args
def do_nothing(x):
    pass

def set_preset(preset):
    """Command-line arguments to the system

    Parameters:
    preset (tuple::<np.array,np.array>): min and max range of form (BGR, BGR) to set trackbars to.
    """    
    cv2.setTrackbarPos(f'min_{FindRangeConstants.BLUE}', FindRangeConstants.CV2_TRACKBAR_WINDOW_NAME, preset[0][0])
    cv2.setTrackbarPos(f'min_{FindRangeConstants.GREEN}', FindRangeConstants.CV2_TRACKBAR_WINDOW_NAME, preset[0][1])
    cv2.setTrackbarPos(f'min_{FindRangeConstants.RED}', FindRangeConstants.CV2_TRACKBAR_WINDOW_NAME, preset[0][2])
    cv2.setTrackbarPos(f'max_{FindRangeConstants.BLUE}', FindRangeConstants.CV2_TRACKBAR_WINDOW_NAME, preset[1][0])
    cv2.setTrackbarPos(f'max_{FindRangeConstants.GREEN}', FindRangeConstants.CV2_TRACKBAR_WINDOW_NAME, preset[1][1])
    cv2.setTrackbarPos(f'max_{FindRangeConstants.RED}', FindRangeConstants.CV2_TRACKBAR_WINDOW_NAME, preset[1][2])

def create_slider_window():
    """Create slider adjustment window.
    """
    cv2.namedWindow('Track Bars', cv2.WINDOW_NORMAL)
    for color in FindRangeConstants.BGR:
        cv2.createTrackbar(f'min_{color}', FindRangeConstants.CV2_TRACKBAR_WINDOW_NAME, 0, 255, do_nothing)
    for color in FindRangeConstants.BGR:
        cv2.createTrackbar(f'max_{color}', FindRangeConstants.CV2_TRACKBAR_WINDOW_NAME, 0, 255, do_nothing)
    cv2.createTrackbar(f'redraw_rate', FindRangeConstants.CV2_TRACKBAR_WINDOW_NAME, 1, 600, do_nothing)
def get_colors(path):
    """Get the preset color ranges for each button color from the presets file.

    Args:
        path (str): .json filepath to read the the values from

    Returns:
        dict::<str, tuple::<np.array,np.array>>: tuple of min and max BGR values to create a new mask from.
    """
    with open(path) as file:
        data = json.load(file)
    colors = {}
    for color in data:
        vals = list(data[color].values())
        colors[color] = (np.array(vals[0:3], dtype=np.uint8), np.array(vals[3:], dtype=np.uint8))
    return colors
def get_all_trackbar_pos():
    """Get current trackbar positions to draw a new masked frame with.

    Returns:
        tuple::<np.array,np.array>>: min and max BGR ranges to draw masked frame with.
    """
    min_blue = cv2.getTrackbarPos(f'min_{FindRangeConstants.BLUE}', FindRangeConstants.CV2_TRACKBAR_WINDOW_NAME)
    min_green = cv2.getTrackbarPos(f'min_{FindRangeConstants.GREEN}', FindRangeConstants.CV2_TRACKBAR_WINDOW_NAME)
    min_red = cv2.getTrackbarPos(f'min_{FindRangeConstants.RED}', FindRangeConstants.CV2_TRACKBAR_WINDOW_NAME)
    max_blue = cv2.getTrackbarPos(f'max_{FindRangeConstants.BLUE}', FindRangeConstants.CV2_TRACKBAR_WINDOW_NAME)
    max_green = cv2.getTrackbarPos(f'max_{FindRangeConstants.GREEN}', FindRangeConstants.CV2_TRACKBAR_WINDOW_NAME)
    max_red = cv2.getTrackbarPos(f'max_{FindRangeConstants.RED}', FindRangeConstants.CV2_TRACKBAR_WINDOW_NAME)
    redraw_rate = cv2.getTrackbarPos('redraw_rate', FindRangeConstants.CV2_TRACKBAR_WINDOW_NAME)
    return (min_blue, min_green, min_red), (max_blue, max_green, max_red), redraw_rate 
if __name__ == '__main__':
    args = _parse_args()
    FILE_PATH = args.presets
    THRESH = args.area_threshold
    redraw_rate = 1
    use_center = True
    use_gaussian = False
    current_color = FindRangeConstants.RED
    colors = get_colors(FILE_PATH)
    preset_state = colors.copy()
    original_colors_from_preset = colors.copy()
    create_slider_window()
    set_preset(colors[current_color])
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
    while True:
        _, frame = cap.read()
        if use_gaussian:
            frame = cv2.GaussianBlur(frame, None, 2)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
        if count % fps == 0:
            mins, maxs, redraw_rate = get_all_trackbar_pos()
            
        colors[current_color] = mins, maxs
        masked = cv2.inRange(hsv_frame, mins, maxs)
        if count % redraw_rate == 0:
            current_contours = []
            for color, range in colors.items():
                contour = find_color(hsv_frame, range, THRESH)
                if contour is not None:
                    current_contours.append((color, contour))
        resultant_frame = np.copy(frame)
        if use_center:
            for (name, (c, cx, cy)) in current_contours:
                resultant_frame = cv2.circle(resultant_frame, (cx, cy), 2, FindRangeConstants.WHITE, 2)
                resultant_frame = cv2.putText(resultant_frame,
                                            name, 
                                            (cx, cy), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 
                                            1, 
                                            color=FindRangeConstants.WHITE, 
                                            thickness=2)
        else:
            for (name, (c, cx, cy)) in current_contours:
                resultant_frame = cv2.drawContours(resultant_frame, [c], -1, FindRangeConstants.WHITE, 3)
                resultant_frame = cv2.putText(resultant_frame, 
                                            name, 
                                            (cx, cy), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 
                                            1, 
                                            color=FindRangeConstants.WHITE, 
                                            thickness=2)
        group_frame[0:HEIGHT, 0:WIDTH] = hsv_frame 
        cv2.putText(group_frame, 'HSV Frame', (WIDTH//2 - WIDTH//8, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color=FindRangeConstants.CYAN, thickness=2)
        group_frame[0:HEIGHT, WIDTH:WIDTH*2] = np.stack([masked, masked, masked], axis=-1)
        cv2.putText(group_frame, f'{current_color}-Masked Frame', (WIDTH + WIDTH//2-WIDTH//4, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color=FindRangeConstants.CYAN, thickness=2)
        group_frame[HEIGHT:HEIGHT*2, 0:WIDTH] = resultant_frame
        cv2.putText(group_frame, 'Contour Applied:', (WIDTH//2-WIDTH//8, HEIGHT + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color=FindRangeConstants.CYAN, thickness=2)
        group_frame[HEIGHT:HEIGHT*2, WIDTH:WIDTH*2] = frame
        cv2.putText(group_frame, 'Original Frame', (WIDTH + WIDTH//2-WIDTH//8, HEIGHT + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color=FindRangeConstants.CYAN, thickness=2)
        cv2.imshow('Capture', group_frame.astype(np.uint8))
        count += 1
        key = cv2.waitKey(25)
        if key == ord('q'):
            break
        elif key == ord('t'):
            use_center = not use_center
        elif key == ord('z'):
            colors = original_colors_from_preset.copy()
            preset_state[current_color] = colors[current_color]
            set_preset(colors[current_color])
        elif key == ord('s'):
            preset_state[current_color] = (mins, maxs)
        elif key == ord('m'):
            use_gaussian = not use_gaussian
        elif key in FindRangeConstants.COLOR_KEYMAPS.keys():
            current_color = FindRangeConstants.COLOR_KEYMAPS[key]
            set_preset(preset_state[current_color])
    #printing the threshold values for usage in detection application
    print(f'min_{FindRangeConstants.BLUE} {mins[0]}  min_{FindRangeConstants.GREEN} {mins[1]} min_{FindRangeConstants.RED} {mins[2]}')
    print(f'max_{FindRangeConstants.BLUE} {maxs[0]}  min_{FindRangeConstants.GREEN} {maxs[1]} min_{FindRangeConstants.RED} {maxs[2]}')
    #destroying all windows
    cv2.destroyAllWindows()
    res = input('Enter filename to save or type q to quit: ')
    if res == 'q':
        pass
    else:
        with open(FILE_PATH) as file:
            data = json.load(file)
            new_json_path = os.path.join(os.getcwd(), 'color-ranges', res)
            with open(new_json_path, 'w') as file:
                for key, range in preset_state.items():
                    data[key]['min_blue'] = int(range[0][0])
                    data[key]['min_green'] = int(range[0][1])
                    data[key]['min_red'] = int(range[0][2])
                    data[key]['max_blue'] = int(range[1][0])
                    data[key]['max_green'] = int(range[1][1])
                    data[key]['max_red'] = int(range[1][2])
                json_object = json.dumps(data, indent=4)
                file.write(json_object)
            print("JSON saved to: ", new_json_path)