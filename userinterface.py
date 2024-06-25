import cv2 as cv
import numpy as np
import os
import dialog
import json

BACKGROUND_COLOR = (0.8, 0.8, 0.8)
FOREGROUND_COLOR = (1., 0.6, 0.6)
FOCUSED_COLOR = (0.4, 0.4, 0.4)

def cross(img, p, color, size):
    s = size // 2
    p00 = [(p[0] - s), (p[1] - s)]
    p01 = [(p[0] + s), (p[1] - s)]
    p10 = [(p[0] - s), (p[1] + s)]
    p11 = [(p[0] + s), (p[1] + s)]
    img = cv.line(img, p00, p11, color, thickness=1)
    img = cv.line(img, p10, p01, color, thickness=1)
    return img

def get_sub_dirs(dir, level, max_level = 3):
    sub_dirs = []
    if level > max_level:
        return sub_dirs
    try:
        for entry in sorted(os.listdir(dir)):
            if os.path.isdir(f'{dir}/{entry}'):
                sub_dirs.append(f'{dir}/{entry}')
                for sub_dir in get_sub_dirs(f'{dir}/{entry}', level + 1, max_level=max_level):
                    sub_dirs.append(sub_dir)
    except:
        pass
    return sub_dirs

class CVAnnoUI(dialog.Window):
    def __init__(self, annotation_dir:str = None):
        super().__init__('CVAnno - Annotation Tool', size = (600, 300))
        if annotation_dir is None:
            ofd = dialog.OpenFolderDialog(annotation_dir)
            self.annotation_dir = ofd.path
        else:
            self.annotation_dir = annotation_dir

        self.load_image_names()
        self.selected_image_idx = 0

        self.txt_annotation_dir = TextBlock(None, self.annotation_dir, px=5, py=30, w=self.size[0] - 100 - 5 - 10)
        self.txt_image_header = TextBlock(None, 'Image', px=5, py=55, align='left', bold=True)
        self.txt_selected_image = TextBlock(None, self.image_names[self.selected_image_idx], px=5+20+5, py=80, w=400, align='center')
        self.ui_elements = [
            # directory
            TextBlock(None, 'Directory', px=5, py=5, align='left', bold=True),
            self.txt_annotation_dir,
            Button('Select folder...', px=self.size[0] - 100 - 10, py = 30, w = 100,\
                   on_left_button_clicked=self.btn_select_folder_clicked),

            # image
            self.txt_image_header,
            Button('<', px=5, py=80, w = 20,\
                on_left_button_clicked=self.btn_previous_image_clicked),
            Button('>', px=5 + 20 + 5 + self.txt_selected_image.w + 5, py=80, w = 20,\
                on_left_button_clicked=self.btn_next_image_clicked),
            self.txt_selected_image,

            # point
            TextBlock(None, 'Point', px=5, py=105, align='left', bold=True),
            Button('<', px=5, py=130, w = 20,\
                on_left_button_clicked=self.btn_previous_point_clicked),
            Button('>', px=5 + 20 + 5 + self.txt_selected_image.w + 5, py=130, w = 20,\
                on_left_button_clicked=self.btn_next_point_clicked),
        ]
        self.load_image()

    def load_image_names(self):
        img_dir = f'{self.annotation_dir}/images'
        self.image_names = []
        for entry in sorted(os.listdir(img_dir)):
            if not entry.lower().endswith('.png') and not entry.lower().endswith('.jpg') and not entry.lower().endswith('.jpeg'):
                continue

            self.image_names.append(entry)

    def load_image(self):
        img_name = self.image_names[self.selected_image_idx]
        self.txt_selected_image.text = img_name
        self.txt_image_header.text = f'Image ({self.selected_image_idx+1}/{len(self.image_names)})'

        id = self.image_names[self.selected_image_idx].split('.')[-2]
        label_file = f'{self.annotation_dir}/labels/{id}.json'
        if os.path.isfile(label_file):
            with open(label_file, 'r') as f:
                self.dict = json.load(f)
        # label file does not exist
        else:
            self.dict = {
                'image_id': img_name,
                'polys': [[[frame.shape[1]//2, frame.shape[0]//2]]]
            }
            max_h, max_w = 1500, 1500
            if frame.shape[0] > max_h:
                scale = max_h / frame.shape[0]
                frame = cv.resize(frame, (int(scale * frame.shape[1]), int(scale * frame.shape[0])))
                cv.imwrite(f'{self.annotation_dir}/images/{img_name}', frame)
            if frame.shape[1] > max_w:
                scale = max_w / frame.shape[1]
                frame = cv.resize(frame, (int(scale * frame.shape[1]), int(scale * frame.shape[0])))
                cv.imwrite(f'{self.annotation_dir}/images/{img_name}', frame)
        pass

    def load_point(self):
        pass

    def btn_previous_image_clicked(self):
        if self.selected_image_idx > 0:
            self.selected_image_idx -= 1
            self.load_image()

    def btn_next_image_clicked(self):
        if self.selected_image_idx < len(self.image_names) - 1:
            self.selected_image_idx += 1
            self.load_image()

    def btn_previous_point_clicked(self):
        pass

    def btn_next_point_clicked(self):
        pass

    def btn_select_folder_clicked(self):
        ofd = dialog.OpenFolderDialog(self.annotation_dir)
        self.annotation_dir = ofd.path
        self.txt_annotation_dir.text = self.annotation_dir
        self.load_image_names()
        self.selected_image_idx = 0
        self.txt_selected_image.text = self.image_names[self.selected_image_idx]
        self.load_image()

class UIElement():
    def __init__(self, label, px, py, w, h, fontScale = 0.4, on_left_button_clicked = None, focusable = True):
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

    def mouse_left_button_clicked(self, x, y):
        if x >= self.px and x <= self.px + self.w and \
            y >= self.py and y <= self.py + self.h:
            self.is_focused = True
            if self.on_left_button_clicked is not None:
                self.on_left_button_clicked()
        else:
            self.is_focused = False
        
        

    def mouse_right_button_clicked(self, x, y):
        pass

    def mouse_moved(self, x, y):
        if x >= self.px and x <= self.px + self.w and \
            y >= self.py and y <= self.py + self.h:
            self.is_mouse_over = True
        else:
            self.is_mouse_over = False

    def key_input(self, key):
        pass

    def draw(self, img):
        if self.focusable and self.is_focused:
            img = cv.rectangle(img, (self.px - self.margin_inner, self.py - self.margin_inner), (self.px + self.w + self.margin_inner, self.py + self.h + self.margin_inner), color=FOCUSED_COLOR, thickness=1)

        if self.focusable and self.is_mouse_over:
            img = cv.rectangle(img, (self.px - self.margin_outer, self.py - self.margin_outer), (self.px + self.w + self.margin_outer, self.py + self.h + self.margin_outer), color=FOREGROUND_COLOR, thickness=1)

class TextBox(UIElement):
    def __init__(self, label, text, px, py, w = 50):
        super().__init__(label, px, py, w, 20)
        self.text = text
        self.enter_pressed = False
        self.cursor_pos = len(self.text)
        self.cursor_blink_interval = 5
        self.render_cycles = 0
        self.cursor = '#'
        self.ctrl_pressed = False

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
                self.enter_pressed = True
                self.is_focused = False
            else:
                self.text = self.text[:self.cursor_pos] + chr(key) + self.text[self.cursor_pos:]
                self.cursor_pos += 1
                self.enter_pressed = False

    def draw(self, img):
        super().draw(img)
        img = cv.rectangle(img, (self.px, self.py), (self.px + self.w, self.py + self.h), color=BACKGROUND_COLOR, thickness=1)
        ((fw,fh), baseline) = cv.getTextSize(
            self.text, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=self.fontScale, thickness=1) # empty string is good enough
        show_cursor = self.render_cycles < self.cursor_blink_interval and \
                        self.is_focused
        out_text = ''
        for i, c in enumerate(self.text):
            if i == self.cursor_pos:
                out_text += self.cursor if show_cursor else ' '
                
            out_text += c
        if (self.cursor_pos == len(self.text) or self.cursor_pos == 0) and self.is_focused:
            out_text += self.cursor if show_cursor else ' '
        img = cv.putText(img, out_text, (self.px, self.py + self.h // 2 + fh // 2), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 0))

        self.render_cycles += 1
        if self.render_cycles >= self.cursor_blink_interval * 2:
            self.render_cycles = 0
        return img

class TextBlock(UIElement):
    def __init__(self, label, text, px, py, w = 50, align = 'left', bold = False):
        super().__init__(label, px, py, w, 20, focusable = False)
        self.text = text
        self.align = align
        self.bold = bold

    def draw(self, img):
        super().draw(img)
        ((fw,fh), baseline) = cv.getTextSize(
            self.text, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=self.fontScale, thickness=1) # empty string is good enough

        thickness = 2 if self.bold else 1

        if self.align == 'center':
            img = cv.putText(img, self.text, (self.px + self.w // 2 - fw // 2, self.py + self.h // 2 + fh // 2),\
                             fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 0), thickness=thickness)
        elif self.align == 'left':
            img = cv.putText(img, self.text, (self.px, self.py + self.h // 2 + fh // 2),\
                             fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 0), thickness=thickness)

        return img

class ToggleSwitch(UIElement):
    def __init__(self, label, px, py):
        super().__init__(label, px, py, 40, 20)

        self.is_checked = False

    def mouse_left_button_clicked(self, x, y):
        super().mouse_left_button_clicked(x, y)
        if x >= self.px and x <= self.px + self.w and \
            y >= self.py and y <= self.py + self.h:
            self.is_checked = not self.is_checked

    def draw(self, img):
        super().draw(img)

        # background
        img = cv.rectangle(img, (self.px + self.h // 2, self.py + 4), (self.px + self.w - self.h // 2, self.py + self.h - 4), color=BACKGROUND_COLOR, thickness=-1)
        img = cv.circle(img, (self.px + self.h // 2, self.py + self.h // 2), radius=self.h // 2 - 3, color=BACKGROUND_COLOR, thickness=-1)
        img = cv.circle(img, (self.px + self.w - self.h // 2, self.py + self.h // 2), radius=self.h // 2 - 3, color=BACKGROUND_COLOR, thickness=-1)

        # foreground
        if self.is_checked:
            img = cv.circle(img, (self.px + self.w - self.h // 2, self.py + self.h // 2), radius=self.h // 2, color=FOREGROUND_COLOR, thickness=-1)
        else:
            img = cv.circle(img, (self.px + self.h // 2, self.py + self.h // 2), radius=self.h // 2, color=FOREGROUND_COLOR, thickness=-1)

        # text
        ((fw,fh), baseline) = cv.getTextSize(
            "", fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=self.fontScale, thickness=1) # empty string is good enough
        #factor = (fh-1) / self.fontScale
        img = cv.putText(img, self.label, (self.px + self.w + 5, self.py + (self.h - fh) // 2 + fh), cv.FONT_HERSHEY_SIMPLEX, fontScale=self.fontScale, color=(0, 0, 0))

        return img
    
class Button(UIElement):
    def __init__(self, label, px, py, w = None, on_left_button_clicked = None):
        super().__init__(label, px, py, 40, 20,\
                         on_left_button_clicked = on_left_button_clicked)

        # text
        ((fw,fh), baseline) = cv.getTextSize(
            self.label, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=self.fontScale, thickness=1) # empty string is good enough
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

        img = cv.rectangle(img, (self.px, self.py), (self.px + self.w, self.py + self.h), BACKGROUND_COLOR, thickness=-1)
        if self.is_checked:
            img = cv.rectangle(img, (self.px, self.py), (self.px + self.w, self.py + self.h), BACKGROUND_COLOR, thickness=-1)
        
        #factor = (fh-1) / self.fontScale
        img = cv.putText(img, self.label, (self.px + self.w // 2 - self.fw // 2, self.py + (self.h - self.fh) // 2 + self.fh),\
                         cv.FONT_HERSHEY_SIMPLEX, fontScale=self.fontScale, color=(0, 0, 0))
        return img