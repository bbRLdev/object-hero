import cv2
#empty function
import argparse
def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='find-camera-values.py')
    parser.add_argument('--im_path', type=str, default='D:\object-hero\WIN_20230409_14_25_08_Pro.jpg', help='image to find the color values of')
    args = parser.parse_args()
    return args
def doNothing(x):
    pass
if __name__ == '__main__':
    args = _parse_args()
    cv2.namedWindow('Track Bars', cv2.WINDOW_NORMAL)

    #creating track bars for gathering threshold values of red green and blue
    cv2.createTrackbar('min_blue', 'Track Bars', 0, 255, doNothing)
    cv2.createTrackbar('min_green', 'Track Bars', 0, 255, doNothing)
    cv2.createTrackbar('min_red', 'Track Bars', 0, 255, doNothing)

    cv2.createTrackbar('max_blue', 'Track Bars', 0, 255, doNothing)
    cv2.createTrackbar('max_green', 'Track Bars', 0, 255, doNothing)
    cv2.createTrackbar('max_red', 'Track Bars', 0, 255, doNothing)

    # reading the image
    # object_image = cv2.imread(args.im_path)

    #resizing the image for viewing purposes
    # resized_image = cv2.resize(object_image,(800, 626))

    # #converting into HSV color model
    # hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

    #showing both resized and hsv image in named windows


    #creating a loop to get the feedback of the changes in trackbars
    # frame = cv2.imread(args.im_path)
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    del cap
    count = 0
    while True: #main loop
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #convertion to HSV
        #reading the trackbar values for thresholds
        min_blue = cv2.getTrackbarPos('min_blue', 'Track Bars')
        min_green = cv2.getTrackbarPos('min_green', 'Track Bars')
        min_red = cv2.getTrackbarPos('min_red', 'Track Bars')
        
        max_blue = cv2.getTrackbarPos('max_blue', 'Track Bars')
        max_green = cv2.getTrackbarPos('max_green', 'Track Bars')
        max_red = cv2.getTrackbarPos('max_red', 'Track Bars')
        #using inrange function to turn on the image pixels where object threshold is matched
        masked = cv2.inRange(hsv_frame, (min_blue, min_green, min_red), (max_blue, max_green, max_red))
        cv2.imshow('reg frame', frame)
        cv2.imshow('hsv', hsv_frame)

            # cv2.imshow('frame', frame)
            #showing the mask image
            # checking if q key is pressed to break out of loop
        cv2.imshow('masked', masked)
        count += 1
        key = cv2.waitKey(25)
        if key == ord('q'):
            break

    #printing the threshold values for usage in detection application
    print(f'min_blue {min_blue}  min_green {min_green} min_red {min_red}')
    print(f'max_blue {max_blue}  max_green {max_green} max_red {max_red}')
    #destroying all windows
    cv2.destroyAllWindows()