import cv2
import hand_tracking as htm
import argparse
import numpy as np
import json
import imutils 
import vgamepad as vg
import itertools
import math
import sys
STRUM = "strum"
RED = "red"
YELLOW = "yellow"
ORANGE = "orange"
BLUE = "blue"
GREEN = "green"
INPUTS = {
    GREEN : vg.XUSB_BUTTON.XUSB_GAMEPAD_A,
    RED : vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
    YELLOW : vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,
    BLUE : vg.XUSB_BUTTON.XUSB_GAMEPAD_X,
    ORANGE : vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER,
    STRUM : vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP
}
def find_color(frame, points):
    mask = cv2.inRange(frame, points[0], points[1])#create mask with boundaries 
    cnts = cv2.findContours(mask, cv2.RETR_TREE, 
                           cv2.CHAIN_APPROX_SIMPLE) # find contours from mask
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        area = cv2.contourArea(c) # find how big countour is
        if area > 1000:       # only if countour is big enough, then
            M = cv2.moments(c)
            cx = int(M['m10'] / M['m00']) # calculate X position
            cy = int(M['m01'] / M['m00']) # calculate Y position
            return c, cx, cy
    return None
def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='main.py')    
    parser.add_argument('--bgr_range_path', type=str, default='color-ranges\color-values-gopro-table.json', help='JSON file to load max and min BGR values for each color')
    parser.add_argument('--draw_buttons_interval', type=int, default=10, help='Reset button contours every X seconds')
    parser.add_argument('--difficulty', type=str, default='easy', help='difficulty level. maps middle 3 right fingers to green, red, and yellow respectively')
    parser.add_argument('--distance', type=float, default=20.0, help='distance finger has to be to box')
    parser.add_argument('--use_center', type=bool, default=False)
    # parser.add_argument('--train_path', type=str, default='data/train.txt', help='path to train set (you should not need to modify)')
    # parser.add_argument('--dev_path', type=str, default='data/dev.txt', help='path to dev set (you should not need to modify)')
    # parser.add_argument('--blind_test_path', type=str, default='data/test-blind.txt', help='path to blind test set (you should not need to modify)')
    # parser.add_argument('--test_output_path', type=str, default='test-blind.output.txt', help='output path for test predictions')
    # parser.add_argument('--no_run_on_test', dest='run_on_test', default=True, action='store_false', help='skip printing output on the test set')
    # parser.add_argument('--word_vecs_path', type=str, default='data/glove.6B.300d-relativized.txt', help='path to word embeddings to use')
    # parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    # parser.add_argument('--feats', type=str, default='UNIGRAM', help='features if using linear model')
    # parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs to train for')
    # parser.add_argument('--hidden_size', type=int, default=100, help='hidden layer size')
    # parser.add_argument('--batch_size', type=int, default=1, help='training batch size; 1 by default and you do not need to batch unless you want to')
    args = parser.parse_args()
    return args
def main():
    gamepad = vg.VX360Gamepad()
    args = _parse_args() 
    cap=cv2.VideoCapture(0)
    DIFFICULTY = args.difficulty
    DISTANCE = args.distance
    # I have defined lower and upper boundaries for each color for my camera
    # Strongly recommended finding for your own camera.
    with open(args.bgr_range_path) as file:
        data = json.load(file)
    colors = {}
    for color in data:
        vals = list(data[color].values())
        colors[color] = (np.array(vals[0:3], dtype=np.uint8), np.array(vals[3:], dtype=np.uint8))
    print(colors)
    detector = htm.handDetector(detectionCon=0.75)
    prev_colors_pressed = set()
    count = 0 
    interval = args.draw_buttons_interval
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(fps)
    while True:
        _,frame=cap.read()
        frame = detector.findHands(frame, draw=True )
        lmList=detector.findPosition(frame,draw=False)
        x_y_list = []
        if(len(lmList) > 1):
            if lmList[8][2]>lmList[5][2]: # pointer
                x_y_list.append((lmList[8][1], lmList[8][2]))
            if lmList[12][2]>lmList[9][2]: # middle
                x_y_list.append((lmList[12][1], lmList[12][2]))
            if lmList[16][2]>lmList[13][2]: # index
                x_y_list.append((lmList[16][1], lmList[16][2]))
            if lmList[20][2]>lmList[17][2]: # pinky
                x_y_list.append((lmList[20][1], lmList[20][2]))
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #convertion to HSV
        # mask =  cv2.inRange(hsv_frame, colors['blue'][0], colors['blue'][1])
        # bitwise = cv2.bitwise_and(mask, motion_frame)

        if count % interval == 0:
            contour_buffer = []
            for name, clr in colors.items(): # for each color in colors
                found_color = find_color(hsv_frame, clr)
                if found_color is not None:
                    c, cx, cy = found_color
                    contour_buffer.append((name, (c, cx, cy)))
                    if not args.use_center:
                        cv2.drawContours(frame, [c], -1, (255, 255, 255), 3) #draw contours
                    else:
                        cv2.circle(frame, (cx, cy), 2, (255, 255, 0), 2)
        else:
            if args.use_center:
                for contour in contour_buffer:
                    cv2.circle(frame, (contour[1][1], contour[1][2]), 2, (255, 255, 255), 2)
            else:
                cv2.drawContours(frame, [contour[1][0] for contour in contour_buffer], -1, (255, 255, 255), 3)
        cur_colors_pressed = set()
        if len(x_y_list) > 0 and len(contour_buffer) > 0:  # call find_color function above
            for x_y, contour in itertools.product(x_y_list, contour_buffer):
                if args.use_center:
                    c, cx, cy = contour[1]
                    x, y = x_y
                    result = math.hypot(cx - x, cy - y)
                else:
                    result = cv2.pointPolygonTest(contour, x_y, True)
                if abs(result) < DISTANCE:
                    cur_colors_pressed.add(contour[0])
                    if args.use_center:
                        print(contour[0], abs(result))
        for color in cur_colors_pressed:
            gamepad.press_button(INPUTS[color])
        gamepad.update()
        colors_to_release = prev_colors_pressed.difference(cur_colors_pressed)
        for color in colors_to_release:
            gamepad.release_button(INPUTS[color])
        gamepad.update()
        print(cur_colors_pressed, prev_colors_pressed)
        prev_colors_pressed = cur_colors_pressed
        count += 1
        cv2.imshow("image",frame)
        if(cv2.waitKey(1) & 0xFF== ord('q')):
            break
if __name__ == "__main__":
    main()