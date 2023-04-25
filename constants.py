import vgamepad as vg

class FindRangeConstants():
    ORANGE = 'orange'
    RED = 'red'
    BLUE = 'blue'
    YELLOW = 'yellow'
    GREEN = 'green'
    WHITE = (255, 255, 255)
    CYAN = (255, 255, 100)
    DEFAULT_HSV_MIN_MAX = [0, 0, 0, 180, 255, 255]
    ALL_COLORS = [GREEN, YELLOW, RED, BLUE, ORANGE]
    CV2_TRACKBAR_WINDOW_NAME = 'Track Bars'
    BGR = [BLUE, GREEN, RED]
    HUE = 'hue'
    SATURATION = 'sat'
    VALUE = 'val'
    HSV = [HUE, SATURATION, VALUE]
    HUE_RANGE = {
        RED: (0, 10),
        ORANGE: (10,20),
        YELLOW: (20, 30),
        GREEN:( 50,70),
        BLUE: (90, 120),
    }
    COLOR_KEYMAPS = {
        ord('r'): RED,
        ord('g'): GREEN,
        ord('b'): BLUE,
        ord('y'): YELLOW,
        ord('o'): ORANGE
    }
class ControllerConstants():
    ORANGE = 'orange'
    RED = 'red'
    BLUE = 'blue'
    YELLOW = 'yellow'
    GREEN = 'green'
    ALL_COLORS = [GREEN, YELLOW, RED, BLUE, ORANGE],
    BGR = [BLUE, GREEN, RED],
    EASY_COLORS = [GREEN, RED, YELLOW]
    WHITE = (255, 255, 255)
    CYAN = (255, 255, 100)
    INPUTS = {
        GREEN : vg.XUSB_BUTTON.XUSB_GAMEPAD_A,
        RED : vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
        YELLOW : vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,
        BLUE : vg.XUSB_BUTTON.XUSB_GAMEPAD_X,
        ORANGE : vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER,
    }
    CV2_TRACKBAR_WINDOW_NAME = 'Track Bars'
    TBAR_TOUCH_THRESH = 'touch_thresh'
    TBAR_CENTER_THRESH = 'center_thresh'
    TBAR_AREA_THRESH = 'area_thresh'
    TBAR_REDRAW_RATE = 'redraw_rate'
    TBAR_INPUT_RATE = 'input_rate'
    
    CV2_TRACKBAR_NAMES = [TBAR_TOUCH_THRESH, TBAR_AREA_THRESH, TBAR_REDRAW_RATE, TBAR_INPUT_RATE]