import cv2 as cv
import os
from datetime import datetime, timedelta

COLOR_LIGHT_GRAY = (0.8, 0.8, 0.8)
COLOR_DARK_GRAY = (0.4, 0.4, 0.4)
COLOR_CORNFLOWER_BLUE = (1., 0.6, 0.6)
FOCUSED_COLOR = (0.4, 0.4, 0.4)
COLOR_FOREGROUND = (0, 0, 0)
COLOR_FOREGROUND_DISABLED = (0.4, 0.4, 0.4)
COLOR_DARKRED = (0., 0., 0.545)

def get_text_size(text:str, fontScale, fontFace = cv.FONT_HERSHEY_SIMPLEX):
    ((fw,fh), baseline) = cv.getTextSize(
            text, fontFace=fontFace, fontScale=fontScale, thickness=1) # empty string is good enough
    return fh, fw

class UIElement():
    def __init__(self, label, px, py, w, h, fontScale = 0.4, on_left_button_clicked = None, focusable = True, is_enabled = True):
        self.label = label
        self.px = px
        self.py = py
        self.w = w
        self.h = h
        self.is_mouse_over = False
        self.margin_inner = 1
        self.margin_outer = 2
        self.fontScale = fontScale
        self.is_focused = False
        self.fontScale = 0.4
        self.on_left_button_clicked = on_left_button_clicked
        self.focusable = focusable
        self.is_enabled = is_enabled

    def mouse_left_button_clicked(self, x, y):
        if not self.is_enabled:
            self.is_focused = False
            self.is_mouse_over = False
            return
        if x >= self.px and x <= self.px + self.w and \
            y >= self.py and y <= self.py + self.h:
            self.is_focused = True
            if self.on_left_button_clicked is not None:
                self.on_left_button_clicked()
        #else:
            #self.is_focused = False
        
    def mouse_right_button_clicked(self, x, y):
        pass

    def mouse_moved(self, x, y):
        if not self.is_enabled:
            self.is_focused = False
            self.is_mouse_over = False
            return
        if x >= self.px and x <= self.px + self.w and \
            y >= self.py and y <= self.py + self.h:
            self.is_mouse_over = True
            if isinstance(self, TextBox):
                pass
        else:
            self.is_mouse_over = False

    def draw(self, img):
        if isinstance(self, TextBox):
            pass
        # focused border
        if self.focusable and self.is_focused:
            img = cv.rectangle(img, (self.px - self.margin_inner, self.py - self.margin_inner), (self.px + self.w + self.margin_inner, self.py + self.h + self.margin_inner), color=FOCUSED_COLOR, thickness=1)

        # mouse over border
        if self.focusable and self.is_mouse_over:
            img = cv.rectangle(img, (self.px - self.margin_outer, self.py - self.margin_outer), (self.px + self.w + self.margin_outer, self.py + self.h + self.margin_outer), color=COLOR_CORNFLOWER_BLUE, thickness=1)

    def key_input(self, key):
        pass

class TextBox(UIElement):
    def __init__(self, label, text, px, py, w = 50, check_valid = None):
        super().__init__(label, px, py, w, 20)
        self.text = text
        self.enter_pressed = False
        self.cursor_pos = len(self.text)
        self.cursor_blink_interval = timedelta(milliseconds=1000)
        self.cursor = '|'
        self.last_blink_time = datetime.now()
        self.check_valid = check_valid
        self.is_valid = True if self.check_valid is None else self.check_valid

    def key_input(self, key):
        super().key_input(key)

        if self.is_focused:
            # backspace
            if key == 127:
                self.text = self.text[:self.cursor_pos-1] + self.text[self.cursor_pos:]
                if self.cursor_pos > 0:
                    self.cursor_pos -= 1
                self.enter_pressed = False
            # delete
            elif key == 40:
                self.text = self.text[:self.cursor_pos] + self.text[self.cursor_pos+1:]
                self.enter_pressed = False
            # arrow left
            elif key == 2:
                self.cursor_pos = self.cursor_pos - 1 if self.cursor_pos > 0 else self.cursor_pos
            # arrow right
            elif key == 3:
                self.cursor_pos = self.cursor_pos + 1 if self.cursor_pos < len(self.text) else self.cursor_pos
            # enter
            elif key == 13:
                if self.is_valid:
                    self.enter_pressed = True
                    self.is_focused = False
            else:
                self.text = self.text[:self.cursor_pos] + chr(key) + self.text[self.cursor_pos:]
                self.cursor_pos += 1
                self.enter_pressed = False

    def draw(self, img):
        super().draw(img)
        img = cv.rectangle(img, (self.px, self.py), (self.px + self.w, self.py + self.h), color=COLOR_LIGHT_GRAY, thickness=1)
        fh, fw = get_text_size(self.text, self.fontScale)
        show_cursor = datetime.now() - self.last_blink_time >= self.cursor_blink_interval // 2
        if datetime.now() - self.last_blink_time >= self.cursor_blink_interval:
            self.last_blink_time = datetime.now()

        self.is_valid = True if self.check_valid is None else self.check_valid(self.text)
         # invalid border
        if not self.is_valid:
            img = cv.rectangle(img, (self.px - self.margin_outer, self.py - self.margin_outer), (self.px + self.w + self.margin_outer, self.py + self.h + self.margin_outer), color=COLOR_DARKRED, thickness=1)

        out_text = ''
        for i, c in enumerate(self.text):
            if i == self.cursor_pos:
                out_text += self.cursor if show_cursor else ''
                
            out_text += c
        if (self.cursor_pos == len(self.text) or self.cursor_pos == 0) and self.is_focused:
            out_text += self.cursor if show_cursor else ' '
        img = cv.putText(img, out_text, (self.px, self.py + self.h // 2 + fh // 2), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 0))

        return img

class TextBlock(UIElement):
    def __init__(self, label, text, px, py, w = 50, align = 'left', bold = False):
        super().__init__(label, px, py, w, 20, focusable = False)
        self.text = text
        self.align = align
        self.bold = bold

    def draw(self, img):
        super().draw(img)
        fh, fw = get_text_size(self.text, self.fontScale)

        thickness = 2 if self.bold else 1

        if self.align == 'center':
            img = cv.putText(img, self.text, (self.px + self.w // 2 - fw // 2, self.py + self.h // 2 + fh // 2),\
                             fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 0), thickness=thickness)
        elif self.align == 'left':
            img = cv.putText(img, self.text, (self.px, self.py + self.h // 2 + fh // 2),\
                             fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 0), thickness=thickness)

        return img

class ToggleSwitch(UIElement):
    def __init__(self, label, px, py, on_left_button_clicked = None):
        super().__init__(label, px, py, 40, 20, on_left_button_clicked=on_left_button_clicked)

        self.is_checked = False

    def mouse_left_button_clicked(self, x, y):
        if x >= self.px and x <= self.px + self.w and \
            y >= self.py and y <= self.py + self.h:
            self.is_checked = not self.is_checked
        super().mouse_left_button_clicked(x, y)

    def draw(self, img):
        super().draw(img)

        # background
        img = cv.rectangle(img, (self.px + self.h // 2, self.py + 4), (self.px + self.w - self.h // 2, self.py + self.h - 4), color=COLOR_LIGHT_GRAY, thickness=-1)
        img = cv.circle(img, (self.px + self.h // 2, self.py + self.h // 2), radius=self.h // 2 - 3, color=COLOR_LIGHT_GRAY, thickness=-1)
        img = cv.circle(img, (self.px + self.w - self.h // 2, self.py + self.h // 2), radius=self.h // 2 - 3, color=COLOR_LIGHT_GRAY, thickness=-1)

        # foreground
        if self.is_checked:
            img = cv.circle(img, (self.px + self.w - self.h // 2, self.py + self.h // 2), radius=self.h // 2, color=COLOR_CORNFLOWER_BLUE, thickness=-1)
        else:
            img = cv.circle(img, (self.px + self.h // 2, self.py + self.h // 2), radius=self.h // 2, color=COLOR_DARK_GRAY, thickness=-1)

        # text
        fh, fw = get_text_size(self.label, self.fontScale)
        #factor = (fh-1) / self.fontScale
        img = cv.putText(img, self.label, (self.px + self.w + 5, self.py + (self.h - fh) // 2 + fh), cv.FONT_HERSHEY_SIMPLEX, fontScale=self.fontScale, color=(0, 0, 0))

        return img
    
class Button(UIElement):
    def __init__(self, label, px, py, w = None, on_left_button_clicked = None):
        super().__init__(label, px, py, 40, 20,\
                         on_left_button_clicked = on_left_button_clicked)

        # text
        fh, fw = get_text_size(label, self.fontScale)
        self.fw = fw
        self.fh = fh
        self.w = fw + 5 if w is None else w

        self.is_checked = False

    def mouse_left_button_clicked(self, x, y):
        super().mouse_left_button_clicked(x, y)
        if x >= self.px and x <= self.px + self.w and \
            y >= self.py and y <= self.py + self.h:
            self.is_checked = not self.is_checked

    def draw(self, img):
        super().draw(img)

        img = cv.rectangle(img, (self.px, self.py), (self.px + self.w, self.py + self.h), COLOR_LIGHT_GRAY, thickness=-1)
        if self.is_checked:
            img = cv.rectangle(img, (self.px, self.py), (self.px + self.w, self.py + self.h), COLOR_LIGHT_GRAY, thickness=-1)
        
        img = cv.putText(img, self.label, (self.px + self.w // 2 - self.fw // 2, self.py + (self.h - self.fh) // 2 + self.fh),\
                         cv.FONT_HERSHEY_SIMPLEX, fontScale=self.fontScale, color=COLOR_FOREGROUND if self.is_enabled else COLOR_FOREGROUND_DISABLED)
        return img