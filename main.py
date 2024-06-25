import cv2 as cv
import arguments
import userinterface

if __name__ == '__main__':
    args = arguments.get_args()

    ui = userinterface.CVAnnoUI()

    while True:
        ui.render()
        cv.waitKey(100)