import vgamepad as vg

class FindRangeConstants():
    ORANGE = 'orange'
    RED = 'red'
    BLUE = 'blue'
    YELLOW = 'yellow'
    GREEN = 'green'
    ALL_COLORS = [GREEN, YELLOW, RED, BLUE, ORANGE]
    CV2_TRACKBAR_WINDOW_NAME = 'Track Bars'
    BGR = [BLUE, GREEN, RED]
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