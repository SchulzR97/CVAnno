import cv2 as cv
import numpy as np
from pathlib import Path
import os

def get_drives_windows():
    drives = [ chr(x) + ":" for x in range(65,91) if os.path.exists(chr(x) + ":") ]
    return drives

def get_drives_osx():
    drives = os.listdir('/Volumes')
    for i, drive in enumerate(drives):
        drives[i] = f'/Volumes/{drive}'
    return drives

class Window():
    def __init__(self, win_name:str, size = (500, 300)):
        self.win_name = win_name
        self.size = size
        self.img = np.ones((size[1], size[0], 3), dtype=np.float32)

        self.ui_elements = []

        cv.imshow(self.win_name, self.img)
        cv.setMouseCallback(self.win_name, self.on_mouse_event, None)

    def render(self):
        self.img = np.ones(self.img.shape)
        for ui_element in self.ui_elements:
            self.img = ui_element.draw(self.img)

        cv.imshow(self.win_name, self.img)
        key = cv.waitKey(100)

        if key != -1:
            for ui_element in self.ui_elements:
                ui_element.key_input(key)

    def on_mouse_event(self, event, x, y, flags, param):
        # mouse left click
        if event == 1:
            for ui_element in self.ui_elements:
                ui_element.mouse_left_button_clicked(x, y)
        # mouse right click
        elif event == 2:
            for ui_element in self.ui_elements:
                ui_element.mouse_right_button_clicked(x, y)
        # mouse movement
        elif event == 0:
            for ui_element in self.ui_elements:
                ui_element.mouse_moved(x, y)
        else:
            pass

class OpenFolderDialog():
    def __init__(self, dir = None, width = 700):
        self.window_name = 'Open folder...'
        self.width = width
        self.line_height = 20
        self.selected_idx = 0

        self.sub_dirs = []
        if dir is None:
            for drive in get_drives_windows():
                self.sub_dirs.append(drive)
            for drive in get_drives_osx():
                self.sub_dirs.append(drive)
        else:
            parent_dir = str(Path(dir).parent)
            self.sub_dirs = self.load_sub_dirs(parent_dir)
            for i, d in enumerate(self.sub_dirs):
                if d == dir:
                    self.selected_idx = i
                    break

        self.parent_dir = self.sub_dirs[self.selected_idx]
        self.enter_pressed = False
        while not self.enter_pressed:
            self.render()
        cv.destroyWindow(self.window_name)

    def render(self):
        self.img = np.ones((10, self.width))
        self.print_line('open folder - arrow right')
        self.print_line('close folder - arrow left')
        self.print_line('next folder - arrow down')
        self.print_line('previous folder - arrow up')
        self.print_line('select folder - enter')
        self.print_line('______________________________________')

        for i, sub_dir in enumerate(self.sub_dirs):
            is_selected = i == self.selected_idx
            self.print_line(sub_dir, is_selected)

        cv.imshow(self.window_name, self.img)
        key = cv.waitKey()
        # arrow up
        if key == 0:
            self.selected_idx = self.selected_idx - 1 if self.selected_idx > 0 else self.selected_idx
        # arrow down
        elif key == 1:
            self.selected_idx = self.selected_idx + 1 if self.selected_idx < len(self.sub_dirs) - 1 else self.selected_idx
        # arrow right
        elif key == 3:
            self.sub_dirs = self.load_sub_dirs(self.sub_dirs[self.selected_idx])
        # arrow left
        elif key == 2:
            self.sub_dirs = self.load_sub_dirs(self.parent_dir)
        # enter
        elif key == 13:
            self.path = self.sub_dirs[self.selected_idx]
            self.enter_pressed = True
        # backspace
        # elif key == 127:
            
        #     pass
        pass

    def load_sub_dirs(self, dir):
        self.parent_dir = str(Path(dir).parent)
        sub_dirs = []
        try:
            for entry in sorted(os.listdir(dir)):
                if os.path.isdir(f'{dir}/{entry}'):
                    sub_dirs.append(f'{dir}/{entry}')
        except:
            pass
        self.selected_idx = 0
        return sub_dirs

    def print_line(self, text, is_highlighted = False):
        new_line = np.ones((self.line_height, self.img.shape[1]))
        if is_highlighted:
            new_line[:, :] = 0.8
        new_line = cv.putText(new_line, text, (10, self.line_height//2+3), cv.FONT_HERSHEY_SIMPLEX, 0.4, color=(0, 0, 0), thickness=1)
        self.img = np.concatenate([self.img, new_line])
