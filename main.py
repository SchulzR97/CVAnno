import cv2 as cv
import arguments
import window
import numpy as np

if __name__ == '__main__':
    args = arguments.get_args()

    ui = window.CVAnnoUI(annotation_dir=args.annotation_dir)

    while True:
        ui.render()
        #cv.waitKey(100)