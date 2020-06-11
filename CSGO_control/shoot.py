import pyautogui

width = 1200
height = 800
monitor = {'top': 80, 'left': 0, 'width': width, 'height': height}


def shoot(mid_x, mid_y):
    x = int(mid_x * width)
    y = int(mid_y * height + height / 9)
    pyautogui.moveTo(x, y)
    pyautogui.click()
