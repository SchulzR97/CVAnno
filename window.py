import cv2 as cv
import numpy as np
import userinterface as ui
from pathlib import Path
import os
import json
import platform
import subprocess
import copy
from enum import Enum

OS_NAMES = {
    'Linux': 'Linux',
    'Mac': 'Darwin',
    'Windows': 'Windows'
}

ANNO_COLOR_DEFAULT = (255, 0, 0)
ANNO_COLOR_SELECTED = (0, 0, 255)

def get_drives_windows():
    drives = [ chr(x) + ":" for x in range(65,91) if os.path.exists(chr(x) + ":") ]
    return drives

def get_drives_osx():
    drives = os.listdir('/Volumes')
    for i, drive in enumerate(drives):
        drives[i] = f'/Volumes/{drive}'
    return drives

class MessageBoxButtons(Enum):
    YES_NO = 1,
    OK_CANCEL = 2,
    OK = 3

class Window():
    def __init__(self, win_name:str, size = (500, 300)):
        self.win_name = win_name
        self.fontScale = 0.4
        self.size = size
        self.img = np.ones((size[1], size[0], 3), dtype=np.float32)
        self.dispose = False
        self.ui_elements = []

        cv.imshow(self.win_name, self.img)
        cv.setMouseCallback(self.win_name, self.on_mouse_event, None)

    def render(self):
        self.img = np.ones(self.img.shape)
        for ui_element in self.ui_elements:
            self.img = ui_element.draw(self.img)

        cv.imshow(self.win_name, self.img)
        # key = cv.waitKey(100)

        # if key != -1:
        #     for ui_element in self.ui_elements:
        #         ui_element.key_input(key)

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
            cv.waitKey(1)
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
            if len(self.sub_dirs) > 0 and os.path.isdir(self.sub_dirs[self.selected_idx]):
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

class TextInputDialog(Window):
    def __init__(self, win_name, text, check_valid = None):
        super().__init__(win_name, size=(500, 80))

        self.txt_header = ui.TextBlock(None, self.win_name, px=5, py=5)
        self.txt_input = ui.TextBox(None, text, px=5, py=30, w=self.size[0]-10, check_valid=check_valid)
        self.txt_input.is_focused = True
        self.btn_cancel = ui.Button('Cancel', px=5, py = 55, w = self.size[0]//2-10, on_left_button_clicked=self.on_btn_cancel_clicked)
        self.btn_save = ui.Button('Save', px=self.size[0]//2+5, py = 55, w = self.size[0]//2-10, on_left_button_clicked=self.on_btn_save_clicked)

        self.ui_elements = [
            self.txt_header,
            self.txt_input,
            self.btn_cancel,
            self.btn_save
        ]

        self.confirmed = False
        
        while not self.confirmed:
            self.render()
            key = cv.waitKey(1)
            self.key_input(key)
            if self.txt_input.enter_pressed:
                self.confirmed = True
        cv.destroyWindow(self.win_name)

    def on_btn_cancel_clicked(self):
        cv.destroyWindow(self.win_name)

    def on_btn_save_clicked(self):
        self.confirmed = True
        cv.destroyWindow(self.win_name)

    def render(self):
        super().render()

    def key_input(self, key):
        if key == -1:
            return
        for ui_element in self.ui_elements:
            ui_element.key_input(key)

class SegmentationUI(Window):
    def __init__(self, annotation_dir:str = None):
        super().__init__('CVAnno - Segmentation', size = (700, 300))

        self.annotation_window = SegmentationWindow(annotation_dir)

        self.create_ui()
        self.load_image()

    def render(self):
        if len(self.annotation_window.image_names) == 0:
            img_name = 'No images'
            self.annotation_window.points = []
            self.annotation_window.frame = np.zeros((1,1,3))
        else:
            img_name = self.annotation_window.image_names[self.annotation_window.selected_image_idx]
        self.txt_selected_image.text = f'{img_name} ({self.annotation_window.frame.shape[1]}, {self.annotation_window.frame.shape[0]})'
        self.txt_image_header.text = f'Image ({self.annotation_window.selected_image_idx+1}/{len(self.annotation_window.image_names)})'

        if len(self.annotation_window.points) == 0:
            self.txt_selected_point.text = 'No points'
        else:
            self.txt_selected_point.text = str(self.annotation_window.points[self.annotation_window.p_i])
        self.txt_point_header.text = f'Point ({self.annotation_window.p_i + 1}/{len(self.annotation_window.points)})'

        max_h, max_w = self.annotation_window.config['max_image_height'], self.annotation_window.config['max_image_width']
        self.btn_resize_image.is_enabled = self.annotation_window.frame.shape[0] > max_h or self.annotation_window.frame.shape[1] > max_w

        super().render()
        self.annotation_window.render()

    def create_ui(self):
        # directory
        self.txt_directory_header = ui.TextBlock(None, 'Directory', px=5, py=5, align='left', bold=True)
        self.txt_annotation_dir = ui.TextBlock(None, self.annotation_window.annotation_dir, px=5, py=30, w=self.size[0] - 150 - 5 - 10)
        self.btn_select_directory = ui.Button('Select directory...', px=self.size[0] - 150 - 10, py = 5, w = 150,\
                                            on_left_button_clicked=self.btn_select_directory_clicked)
        self.btn_open_directory = ui.Button('Open directory...', px=self.size[0] - 150 - 10, py = 30, w = 150,\
                                            on_left_button_clicked=self.btn_open_folder_clicked)

        # image
        self.txt_image_header = ui.TextBlock(None, 'Image', px=5, py=80, align='left', bold=True)
        self.btn_rename = ui.Button('Rename', px=self.size[0] - 150 - 10, py = 80, w = 150,\
                                            on_left_button_clicked=self.btn_rename_clicked)
        self.btn_previous_image = ui.Button('<', px=5, py=105, w = 20,\
                                            on_left_button_clicked=self.btn_previous_image_clicked)
        self.txt_selected_image = ui.TextBlock(None, self.annotation_window.image_names[self.annotation_window.selected_image_idx],\
                                            px=5+20+5, py=105, w=500, align='center')
        self.btn_next_image = ui.Button('>', px=5 + 20 + 5 + self.txt_selected_image.w + 5, py=105, w = 20,\
                                            on_left_button_clicked=self.btn_next_image_clicked)
        self.switch_not_annotated_images = ui.ToggleSwitch('Not annotated', px=5+20+5+self.txt_selected_image.w+5+20+5, py=105,\
                                                           on_left_button_clicked=self.switch_not_annotated_images_clicked)
        self.switch_not_annotated_images.is_checked = True
        self.btn_resize_image = ui.Button('Resize', px=self.size[0] - 150 - 10, py = 130, w = 150,\
                                            on_left_button_clicked=self.btn_resize_image_clicked)
        self.btn_save_segmented_image = ui.Button('Save segmented image', px=self.size[0] - 150 - 10, py = 155, w = 150,\
                                            on_left_button_clicked=self.btn_save_segmented_image_clicked)

        # point
        self.txt_point_header = ui.TextBlock(None, 'Point', px=5, py=155, align='left', bold=True)
        self.btn_previous_point = ui.Button('<', px=5, py=180, w = 20,\
                                            on_left_button_clicked=self.btn_previous_point_clicked)
        self.txt_selected_point = ui.TextBlock(None, str(self.annotation_window.points[self.annotation_window.p_i]),\
                                            px=5+20+5, py=180, w=400, align='center')
        self.btn_next_point = ui.Button('>', px=5 + 20 + 5 + self.txt_selected_image.w + 5, py=180, w = 20,\
                                        on_left_button_clicked=self.btn_next_point_clicked)
        
        # quit
        self.btn_quit = ui.Button('Quit', px=self.size[0] - 10 - 50, py=self.size[1] - 10 - 20, w=50, on_left_button_clicked=self.btn_quit_clicked)

        self.ui_elements = [
            # directory
            self.txt_directory_header,
            self.txt_annotation_dir,
            self.btn_select_directory,
            self.btn_open_directory,

            # image
            self.txt_image_header,
            self.btn_rename,
            self.btn_previous_image,
            self.btn_next_image,
            self.txt_selected_image,
            self.switch_not_annotated_images,
            self.btn_resize_image,
            self.btn_save_segmented_image,

            # point
            self.txt_point_header,
            self.btn_previous_point,
            self.txt_selected_point,
            self.btn_next_point,

            # quit
            self.btn_quit
        ]

    def load_image(self):
        self.annotation_window.load_image()

#region UI interaction
    def btn_rename_clicked(self):
        def check_valid(text):
            for img_name in self.annotation_window.image_names:
                if img_name.split('.')[-2] == text:
                    return False
            if len(text) == 0:
                return False
            return True

        img_name = self.annotation_window.image_names[self.annotation_window.selected_image_idx]
        appendix = img_name.split('.')[-1]
        id = img_name.split('.')[-2]
        txt_input = TextInputDialog('Rename annotation', id, check_valid=check_valid)

        if txt_input.confirmed:
            new_id = f'{txt_input.txt_input.text}'
            self.annotation_window.image_names[self.annotation_window.selected_image_idx] = f'{new_id}.{appendix}'
            #self.txt_selected_image.text = f'{txt_input.text}.{appendix}'
            #self.render()
            os.rename(f'{self.annotation_window.annotation_dir}/images/{id}.{appendix}', f'{self.annotation_window.annotation_dir}/images/{new_id}.{appendix}')
            os.rename(f'{self.annotation_window.annotation_dir}/labels/{id}.json', f'{self.annotation_window.annotation_dir}/labels/{new_id}.json')

        pass

    def btn_previous_image_clicked(self):
        self.annotation_window.select_previous_image()
        # if self.annotation_window.selected_image_idx == 0:
        #     self.annotation_window.selected_image_idx = len(self.annotation_window.image_names) - 1
        # else:
        #     self.annotation_window.selected_image_idx -= 1
        self.load_image()

    def btn_next_image_clicked(self):
        self.annotation_window.select_next_image()
        # if self.annotation_window.selected_image_idx == len(self.annotation_window.image_names) - 1:
        #     self.annotation_window.selected_image_idx = 0
        # else:
        #     self.annotation_window.selected_image_idx += 1
        self.load_image()

    def switch_not_annotated_images_clicked(self):
        self.annotation_window.load_image_names(self.switch_not_annotated_images.is_checked)
        self.load_image()
        #self.render()

    def btn_resize_image_clicked(self):
        max_h, max_w = self.annotation_window.config['max_image_height'], self.annotation_window.config['max_image_width']
        scale = np.min([max_h / self.annotation_window.frame.shape[0], max_w / self.annotation_window.frame.shape[1]])
        frame = cv.resize(self.annotation_window.frame, (int(np.round(scale * self.annotation_window.frame.shape[1])), int(np.round(scale * self.annotation_window.frame.shape[0]))))
        points = copy.copy(self.annotation_window.points)

        if scale == 1:
            return

        for i, p in enumerate(self.annotation_window.points):
            self.annotation_window.points[i][0] = int(np.round(scale * p[0]))
            self.annotation_window.points[i][1] = int(np.round(scale * p[1]))
            pass
        
        fname = f'{self.annotation_window.annotation_dir}/images/{self.annotation_window.image_names[self.annotation_window.selected_image_idx]}'
        if cv.imwrite(fname, frame):
            self.annotation_window.points = points
            self.annotation_window.frame = frame
            self.annotation_window.pending_changes = True
        pass

    def btn_save_segmented_image_clicked(self):
        segmented_dir = f'{self.annotation_window.annotation_dir}/segmented'
        if not os.path.isdir(segmented_dir):
            os.mkdir(segmented_dir)

        fname = f'{self.annotation_window.annotation_dir}/images/{self.annotation_window.image_names[self.annotation_window.selected_image_idx]}'
        img = cv.imread(fname)
        
        overlay = np.zeros((img.shape[0], img.shape[1]))
        points = np.array(self.annotation_window.points)
        overlay = cv.fillPoly(overlay, [points], (255, 255, 255))

        png = np.full((img.shape[0], img.shape[1], 4), 255, dtype=np.uint8)
        png[:, :, 0:3] = img.copy()
        png[:, :, 3] = overlay
        full_name = self.annotation_window.image_names[self.annotation_window.selected_image_idx]
        fname = full_name.split('.')[-2]
        cv.imwrite(f'{segmented_dir}/{fname}.png', png)

    def btn_previous_point_clicked(self):
        if self.annotation_window.p_i == 0:
            self.annotation_window.p_i = len(self.annotation_window.points) - 1
        else:
            self.annotation_window.p_i -= 1
        self.txt_selected_point.text = str(self.annotation_window.points[self.annotation_window.p_i])
        self.txt_point_header.text = f'Point ({self.annotation_window.p_i + 1}/{len(self.annotation_window.points)})'

    def btn_next_point_clicked(self):
        if self.annotation_window.p_i == len(self.annotation_window.points) - 1:
            self.annotation_window.p_i = 0
        else:
            self.annotation_window.p_i += 1
        self.txt_selected_point.text = str(self.annotation_window.points[self.annotation_window.p_i])
        self.txt_point_header.text = f'Point ({self.annotation_window.p_i + 1}/{len(self.annotation_window.points)})'

    def btn_select_directory_clicked(self):
        ofd = OpenFolderDialog(self.annotation_window.annotation_dir)
        self.annotation_window.annotation_dir = ofd.path
        self.txt_annotation_dir.text = self.annotation_window.annotation_dir
        self.annotation_window.load_image_names(self.switch_not_annotated_images.is_checked)
        self.annotation_window.selected_image_idx = 0
        self.annotation_window.load_config()
        self.load_image()

    def btn_open_folder_clicked(self):
        if platform.system() == OS_NAMES['Mac']:
            #subprocess.Popen(["finder", self.annotation_dir])
            subprocess.call(["open", "-R", f'{self.annotation_window.annotation_dir}'])
        else:
            raise Exception(f'Open folder is not implemented for {platform.system()}.')

    def btn_quit_clicked(self):
        self.dispose = True
#endregion

class SegmentationWindow(Window):
    def __init__(self, annotation_dir:str):
        super().__init__('CVAnno - Segmentation Window')

        if annotation_dir is None:
            ofd = OpenFolderDialog(annotation_dir)
            self.annotation_dir = ofd.path
        else:
            self.annotation_dir = annotation_dir

        self.load_config()
        self.load_image_names()

        self.selected_image_idx = 0
        self.frame = np.ones((10, 10, 3))
        self.points = [[-1, -1]]
        self.p_i = 0
        self.pending_changes = False

        self.render()
        cv.setMouseCallback(self.win_name, self.on_click, None)

    def on_click(self, event, x, y, flags, param):
        t1 = cv.EVENT_LBUTTONDBLCLK
        if event == 1:
            self.points[self.p_i][0] = x
            self.points[self.p_i][1] = y
            self.pending_changes = True

    def handle_key_input(self):
        key = cv.waitKey(100)

        if key == 0:    # UP
            self.points[self.p_i][1] -= 1
            self.pending_changes = True
        elif key == 2:  # LEFT
            self.points[self.p_i][0] -= 1
            self.pending_changes = True
        elif key == 1:  # DOWN
            self.points[self.p_i][1] += 1
            self.pending_changes = True
        elif key == 3:  # RIGHT
            self.points[self.p_i][0] += 1
            self.pending_changes = True
        elif key == ord('+'):
            self.add_point()
            self.select_next_point()
            self.pending_changes = True
        elif key == ord('-'):
            self.remove_point()
            self.select_previous_point()
            self.pending_changes = True
        elif key == ord('1'):  # previous point
            self.select_previous_point()
        elif key == ord('2'):  # next point
            self.select_next_point()
        elif key == ord('s'):   # SAVE
            self.save_annotations()
        elif key == ord('p'):
            self.select_previous_image()
            self.load_image()
        elif key == ord('n'):
            self.select_next_image()
            self.load_image()

    def load_config(self):
        config_filename = f'{self.annotation_dir}/config.json'
        if os.path.isfile(config_filename):
            with open(config_filename, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                'max_image_width': 1500,
                'max_image_height': 1500
            }
            with open(config_filename, 'w+') as f:
                json.dump(self.config, f)

    def load_image_names(self, load_not_annotated_images:bool = True):
        img_dir = f'{self.annotation_dir}/images'
        self.image_names = []
        for entry in sorted(os.listdir(img_dir)):
            if not entry.lower().endswith('.png') and not entry.lower().endswith('.jpg') and not entry.lower().endswith('.jpeg'):
                continue

            if not load_not_annotated_images:
                id = entry.split('.')[-2]
                label_file = f'{self.annotation_dir}/labels/{id}.json'
                if os.path.isfile(label_file):
                    continue

            self.image_names.append(entry)
        self.selected_image_idx = 0
        pass

    def load_image(self):
        if len(self.image_names) == 0:
            img_name = 'No images'
            return
        else:
            img_name = self.image_names[self.selected_image_idx]
        
        id = img_name.split('.')[-2]
        label_file = f'{self.annotation_dir}/labels/{id}.json'
        if os.path.isfile(label_file):
            with open(label_file, 'r') as f:
                self.dict = json.load(f)
        # label file does not exist
        else:
            self.dict = {
                'image_id': img_name,
                'polys': [[[self.frame.shape[1]//2, self.frame.shape[0]//2]]]
            }
            max_h, max_w = self.config['max_image_height'], self.config['max_image_width']
            self.frame = cv.imread(f'{self.annotation_dir}/images/{img_name}')
            if self.frame.shape[0] > max_h:
                scale = max_h / self.frame.shape[0]
                self.frame = cv.resize(self.frame, (int(scale * self.frame.shape[1]), int(scale * self.frame.shape[0])))
                cv.imwrite(f'{self.annotation_dir}/images/{img_name}', self.frame)
            if self.frame.shape[1] > max_w:
                scale = max_w / self.frame.shape[1]
                self.frame = cv.resize(self.frame, (int(scale * self.frame.shape[1]), int(scale * self.frame.shape[0])))
                cv.imwrite(f'{self.annotation_dir}/images/{img_name}', self.frame)
        self.load_points()
        self.p_i = 0
        self.frame = cv.imread(f'{self.annotation_dir}/images/{img_name}')

    def load_points(self):
        self.points = self.dict['polys'][0]

    def render(self):
        img = copy.copy(self.frame)

        if len(self.points) >= 3:
            img = self.poly(img, self.points, color=ANNO_COLOR_DEFAULT)

        for i, point in enumerate(self.points):
            color = ANNO_COLOR_SELECTED if i == self.p_i else ANNO_COLOR_DEFAULT

            max_shape = np.max([img.shape[0], img.shape[1]])

            if i == self.p_i:
                size = int(np.round(0.005 * max_shape))
                img = self.cross(img, self.points[i], color, size)
            else:
                radius = int(np.round(0.002 * max_shape))
                img = self.circle(img, self.points[i], color, radius=radius)
            
            img = self.line(img, self.points[i-1], self.points[i], color)

        cv.imshow(self.win_name, img)
        self.handle_key_input()

    def select_next_image(self):
        if self.pending_changes:
            if self.show_save_messageBox():
                self.save_annotations()
        if self.selected_image_idx < len(self.image_names) - 1:
            self.selected_image_idx += 1
        else:
            self.selected_image_idx = 0

    def select_previous_image(self):
        if self.pending_changes:
            if self.show_save_messageBox():
                self.save_annotations()
        if self.selected_image_idx == 0:
            self.selected_image_idx = len(self.image_names) - 1
        else:
            self.selected_image_idx -= 1

    def add_point(self):
        self.points.insert(self.p_i, [self.points[self.p_i][0], self.points[self.p_i][1]])

    def remove_point(self):
        if len(self.points) > 1:
            self.points.pop(self.p_i)

    def select_next_point(self):
        if self.p_i < len(self.points) - 1:
            self.p_i += 1
        else:
            self.p_i = 0

    def select_previous_point(self):
        if self.p_i > 0:
            self.p_i -= 1
        else:
            self.p_i = len(self.points) - 1

    def save_annotations(self):
        self.dict['polys'] = [self.points]

        id = self.image_names[self.selected_image_idx].split('.')[-2]
        filename = f'{self.annotation_dir}/labels/{id}.json'
        with open(filename, 'w') as f:
            json.dump(self.dict, f)
        self.pending_changes = False

    def poly(self, img, points, color):
        overlay = img.copy() 
    
        points = np.array(points)
        overlay = cv.fillPoly(overlay, [points], color) 
        
        alpha = 0.4
        img = cv.addWeighted(overlay, alpha, img, 1 - alpha, 0) 

        return img
    
    def line(self, img, p1, p2, color):
        img = cv.line(img, p1, p2, color=color, thickness=1)
        return img

    def circle(self, img, point, color, radius = 3):
        img = cv.circle(img, point, radius=radius, color=color, thickness=-1)
        return img

    def cross(self, img, p, color, size):
        s = size // 2
        p00 = [(p[0] - s), (p[1] - s)]
        p01 = [(p[0] + s), (p[1] - s)]
        p10 = [(p[0] - s), (p[1] + s)]
        p11 = [(p[0] + s), (p[1] + s)]
        img = cv.line(img, p00, p11, color, thickness=3)
        img = cv.line(img, p10, p01, color, thickness=3)
        return img

    def show_save_messageBox(self):
        msg = MessageBox('Pending changes', 'There are pending changes. Should they be saved?', MessageBoxButtons.YES_NO)
        if msg.accept:
            self.save_annotations()

class MessageBox(Window):
    def __init__(self, caption:str, message:str, buttons:MessageBoxButtons):
        super().__init__(caption)
        fh, fw = ui.get_text_size(message, self.fontScale)
        self.size = (fw + 10, fh + 10 + 40)
        self.img = np.ones((self.size[1], self.size[0], 3), dtype=np.float32)

        self.ui_elements = [
            ui.TextBlock(None, message, px=5, py=5, align='left', bold=False)
        ]

        if buttons == MessageBoxButtons.YES_NO:
            self.ui_elements.append(ui.Button('Yes', px=5, py=5+fh+20, w=self.size[0]//2-10, on_left_button_clicked=self.on_accept))
            self.ui_elements.append(ui.Button('No', px=self.size[0]//2 + 5, py=5+fh+20, w=self.size[0]//2-10, on_left_button_clicked=self.on_decline))
        elif buttons == MessageBoxButtons.OK_CANCEL:
            self.ui_elements.append(ui.Button('Ok', px=5, py=5+fh+20, w=self.size[0]//2-10, on_left_button_clicked=self.on_accept))
            self.ui_elements.append(ui.Button('Cancel', px=self.size[0]//2 + 5, py=5+fh+20, w=self.size[0]//2-10, on_left_button_clicked=self.on_decline))
        elif buttons == MessageBoxButtons.OK:
            self.ui_elements.append(ui.Button('Ok', px=5, py=5+fh+20, w=self.size[0]-10, on_left_button_clicked=self.on_accept))

        self.render()

        self.accept = None

        while self.accept is None:
            cv.waitKey(1)
        cv.destroyWindow(self.win_name)

    def on_decline(self):
        self.accept = False

    def on_accept(self):
        self.accept = True
        

    # def render(self):
    #     super().render()
    #     cv.waitKey(10)