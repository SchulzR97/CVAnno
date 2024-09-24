import cv2 as cv
import arguments
import window
import model

if __name__ == '__main__':
    args = arguments.get_args()

    unet = model.load_unet('/Users/schulzr/Documents/GIT/depthmap-action-prediction/runs/segmentation_20240705115549/model.pt', grayscale=True)
    ui = window.SegmentationUI(annotation_dir=args.annotation_dir, unet=unet)

    while not ui.dispose:
        ui.render()
        #cv.waitKey(100)