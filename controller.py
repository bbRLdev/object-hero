import cv2
import hand_tracking as htm
import argparse
import numpy as np
import json
import vgamepad as vg
import itertools
import math
from constants import ControllerConstants
from utils import find_color


def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='main.py')    
    parser.add_argument('--color_range_path', type=str, default='./color-ranges/hsv-test.json', help='JSON file to load max and min HSV values for each color')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args

def do_nothing():
    pass
def create_slider_window():
    """Create slider adjustment window.
    """
    cv2.namedWindow('Track Bars', cv2.WINDOW_NORMAL)
    cv2.createTrackbar(ControllerConstants.TBAR_TOUCH_THRESH, ControllerConstants.CV2_TRACKBAR_WINDOW_NAME, 100, 300, do_nothing)
    cv2.createTrackbar(ControllerConstants.TBAR_AREA_THRESH,  ControllerConstants.CV2_TRACKBAR_WINDOW_NAME, 1000, 2000, do_nothing)
    cv2.createTrackbar(ControllerConstants.TBAR_CENTER_THRESH,  ControllerConstants.CV2_TRACKBAR_WINDOW_NAME, 1, 100, do_nothing)
    cv2.createTrackbar(ControllerConstants.TBAR_INPUT_RATE,  ControllerConstants.CV2_TRACKBAR_WINDOW_NAME, 1, 150, do_nothing)
    cv2.createTrackbar(ControllerConstants.TBAR_REDRAW_RATE,  ControllerConstants.CV2_TRACKBAR_WINDOW_NAME, 1, 150, do_nothing)

def get_all_slider_pos():
    """Get current trackbar positions to draw a new masked frame with.

    Returns:
        tuple::<np.array,np.array>>: min and max BGR ranges to draw masked frame with.
    """
    touch_thresh = cv2.getTrackbarPos(ControllerConstants.TBAR_TOUCH_THRESH, ControllerConstants.CV2_TRACKBAR_WINDOW_NAME)
    area_thresh = cv2.getTrackbarPos(ControllerConstants.TBAR_AREA_THRESH, ControllerConstants.CV2_TRACKBAR_WINDOW_NAME)
    center_thresh = cv2.getTrackbarPos(ControllerConstants.TBAR_CENTER_THRESH, ControllerConstants.CV2_TRACKBAR_WINDOW_NAME)
    input_rate = cv2.getTrackbarPos(ControllerConstants.TBAR_INPUT_RATE, ControllerConstants.CV2_TRACKBAR_WINDOW_NAME)
    redraw_rate = cv2.getTrackbarPos(ControllerConstants.TBAR_REDRAW_RATE, ControllerConstants.CV2_TRACKBAR_WINDOW_NAME)
    return touch_thresh, area_thresh, center_thresh, input_rate, redraw_rate 
def main():
    gamepad = vg.VX360Gamepad()
    create_slider_window()
    args = _parse_args() 
    cap=cv2.VideoCapture(0)
    touch_thresh, area_thresh, center_thresh, input_interval, draw_interval = get_all_slider_pos()
    VERBOSE = args.verbose
    REFRESH_TRACK_BAR_INTERVAL = 30
    use_center = True
    use_gaussian = False
    # I have defined lower and upper boundaries for each color for my camera
    # Strongly recommended finding for your own camera.
    with open(args.color_range_path) as file:
        data = json.load(file)
    colors = {}
    for color in data:
        vals = list(data[color].values())
        colors[color] = (np.array(vals[0:3], dtype=np.uint8), np.array(vals[3:], dtype=np.uint8))

    detector = htm.HandDetector(detection_confidence=0.75)
    prev_colors_pressed = set()
    count = 0 
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if VERBOSE:
        print("Active colors: \n", colors)
        print("Capture FPS: ", fps)
        print("Press the 'm' key to toggle gaussian blur")
        print("Press the 't' key to toggle whether we use centerpt-fingertip distance strategy")
        print("Press the 'q' key to quit")
    while True:
        _,frame=cap.read()
        if use_gaussian:
            frame = cv2.GaussianBlur(frame, None, 2)
        frame, lm_list = detector.find_hands(frame, draw_complex=True)
        x_y_list = []
        for hand in lm_list:
            if(len(hand) > 1):
                # compare x y of finger tip to corresponding knuckle.
                if hand[8][2]>hand[5][2]: # pointer
                    x_y_list.append((hand[8][1], hand[8][2]))
                if hand[12][2]>hand[9][2]: # middle
                    x_y_list.append((hand[12][1], hand[12][2]))
                if hand[16][2]>hand[13][2]: # index
                    x_y_list.append((hand[16][1], hand[16][2]))
                if hand[20][2]>hand[17][2]: # pinky
                    x_y_list.append((hand[20][1], hand[20][2]))
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #convertion to HSV
        if count % draw_interval == 0:
            contour_buffer = []
            for name, clr in colors.items(): # for each color in colors
                found_color = find_color(hsv_frame, clr, area_thresh)
                if found_color is not None:
                    c, cx, cy = found_color
                    contour_buffer.append((name, (c, cx, cy)))
                    if use_center:
                        cv2.circle(frame, (cx, cy), 2, ControllerConstants.CYAN, 2)
                    cv2.drawContours(frame, [c], -1, ControllerConstants.CYAN, 3) #draw contours
        else:
            if use_center:
                for contour in contour_buffer:
                    cv2.circle(frame, (contour[1][1], contour[1][2]), 2, ControllerConstants.WHITE, 2)
            cv2.drawContours(frame, [contour[1][0] for contour in contour_buffer], -1, ControllerConstants.WHITE, 3)
        if count % input_interval == 0:
            cur_colors_pressed = set()
        if len(x_y_list) > 0 and len(contour_buffer) > 0:  # call find_color function above
            for x_y, contour in itertools.product(x_y_list, contour_buffer):
                if contour[0] not in cur_colors_pressed and use_center:
                    c, cx, cy = contour[1]
                    x, y = x_y
                    center_result = np.sqrt(np.square(cx - x) + np.square(cy-y))
                    result = cv2.pointPolygonTest(contour[1][0], x_y, True)
                elif contour[0] not in cur_colors_pressed:
                    result = cv2.pointPolygonTest(contour[1][0], x_y, True)
                if contour[0] not in cur_colors_pressed and abs(result) < touch_thresh:
                    if not use_center:
                        if VERBOSE and contour[0] == ControllerConstants.RED: # report distance of finger tip between red bounding box
                            cv2.putText(frame, f"{ControllerConstants.RED}: {abs(result)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color=ControllerConstants.CYAN, thickness=2)
                        cur_colors_pressed.add(contour[0])
                        gamepad.press_button(ControllerConstants.INPUTS[contour[0]])
                        gamepad.update()
                    elif use_center and center_result < center_thresh:
                        if VERBOSE and contour[0] == ControllerConstants.RED: # report distance of finger tip between red bounding box
                            cv2.putText(frame, f"{ControllerConstants.RED}: {abs(result)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color=ControllerConstants.CYAN, thickness=2)
                        cur_colors_pressed.add(contour[0])
                        gamepad.press_button(ControllerConstants.INPUTS[contour[0]])
                        gamepad.update()
        if VERBOSE:
            cv2.putText(frame, str(cur_colors_pressed), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color=ControllerConstants.CYAN, thickness=2)
            cv2.putText(frame, f"gaussian:{use_gaussian}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, color=ControllerConstants.CYAN, thickness=2)
            cv2.putText(frame, f"centerpt:{use_center}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, color=ControllerConstants.CYAN, thickness=2)
        colors_to_release = prev_colors_pressed.difference(cur_colors_pressed)
        for color in colors_to_release:
            gamepad.release_button(ControllerConstants.INPUTS[color])
            gamepad.update()
        prev_colors_pressed = cur_colors_pressed
        count += 1
        cv2.imshow("image",frame)
        key = cv2.waitKey(20)
        if count % REFRESH_TRACK_BAR_INTERVAL:
            touch_thresh, area_thresh, center_thresh, input_interval, draw_interval = get_all_slider_pos()
        if key == ord('q'):
            break
        elif key == ord('m'):
            use_gaussian = not use_gaussian
        elif key == ord('t'):
            use_center = not use_center
if __name__ == "__main__":
    main()