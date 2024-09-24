import cv2 as cv
import arguments
import window

if __name__ == '__main__':
    args = arguments.get_args()

    ui = window.SegmentationUI(annotation_dir=args.annotation_dir)

    while not ui.dispose:
        ui.render()
        #cv.waitKey(100)