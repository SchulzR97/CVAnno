import cv2 as cv
import numpy as np

def cross(img, p, color, size):
    s = size // 2
    p00 = [(p[0] - s), (p[1] - s)]
    p01 = [(p[0] + s), (p[1] - s)]
    p10 = [(p[0] - s), (p[1] + s)]
    p11 = [(p[0] + s), (p[1] + s)]
    img = cv.line(img, p00, p11, color, thickness=1)
    img = cv.line(img, p10, p01, color, thickness=1)
    return img

class CVAnnoUI():
    def __init__(self, size = (500, 500)):
        self.win_name = 'CVAnno - Annotation Tool'

        self.img = np.ones((size[0], size[1], 3), dtype=np.float32)
        cv.imshow(self.win_name, self.img)
        cv.setMouseCallback(self.win_name, self.on_click, None)

        self.mouse = {
            'clicked': False,
            'x': -1,
            'y': -1
        }

        self.switches = [
            ToggleSwitch('Switch 1', px=5, py=5),
            ToggleSwitch('Switch 2', px=5, py=30)
        ]

        self.render()

    def render(self):
        self.img = np.ones(self.img.shape)
        for btn in self.switches:
            self.img = btn.draw(self.img)

        cv.imshow(self.win_name, self.img)

    def on_click(self, event, x, y, flags, param):
        if event == 1:
            for switch in self.switches:
                switch.win_clicked(x, y)

class ToggleSwitch():
    def __init__(self, text, px, py):
        self.text = text
        self.px = px
        self.py = py
        self.w = 40
        self.h = 20

        self.background_color = (0.8, 0.8, 0.8)
        self.foreground_color = (1., 0.6, 0.6)

        self.checked = False

    def win_clicked(self, x, y):
        if x >= self.px and x <= self.px + self.w and \
            y >= self.py and y <= self.py + self.h:
            self.checked = not self.checked

    def draw(self, img):
        img = cv.rectangle(img, (self.px + self.h // 2, self.py + 4), (self.px + self.w - self.h // 2, self.py + self.h - 4), color=self.background_color, thickness=-1)
        img = cv.circle(img, (self.px + self.h // 2, self.py + self.h // 2), radius=self.h // 2 - 3, color=self.background_color, thickness=-1)
        img = cv.circle(img, (self.px + self.w - self.h // 2, self.py + self.h // 2), radius=self.h // 2 - 3, color=self.background_color, thickness=-1)

        if self.checked:
            img = cv.circle(img, (self.px + self.w - self.h // 2, self.py + self.h // 2), radius=self.h // 2, color=self.foreground_color, thickness=-1)
        else:
            img = cv.circle(img, (self.px + self.h // 2, self.py + self.h // 2), radius=self.h // 2, color=self.foreground_color, thickness=-1)

        fontScale = 0.4
        ((fw,fh), baseline) = cv.getTextSize(
            "", fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, thickness=1) # empty string is good enough
        factor = (fh-1) / fontScale
        img = cv.putText(img, self.text, (self.px + self.w + 5, self.py + (self.h - fh) // 2 + fh), cv.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=(0, 0, 0))

        return img