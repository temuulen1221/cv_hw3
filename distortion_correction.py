import cv2 as cv
import numpy as np

# The given video and calibration data
video_file = '../data/chessboard.avi'
K = np.array([[432.7390364738057, 0, 476.0614994349778],
              [0, 431.2395555913084, 288.7602152621297],
              [0, 0, 1]])  # Camera matrix from calibration
dist_coeff = np.array([-0.2852754904152874, 0.1016466459919075, -0.0004420196146339175, 
                       0.0001149909868437517, -0.01803978785585194])  # Distortion coefficients

# Open the video
video = cv.VideoCapture(video_file)
if not video.isOpened():
    print("Error: Could not open video file.")
    exit()

# Run distortion correction
show_rectify = True  
map1, map2 = None, None

while True:
    # Read an image from the video
    valid, img = video.read()
    if not valid:
        break

    # Rectify geometric distortion
    info = "Original"
    if show_rectify:
        if map1 is None or map2 is None:
            map1, map2 = cv.initUndistortRectifyMap(K, dist_coeff, None, K, (img.shape[1], img.shape[0]), cv.CV_32FC1)
        img = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR)
        info = "Rectified"

    cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
    cv.imshow('Distortion Correction', img)

    key = cv.waitKey(30) & 0xFF
    if key == 32:  # Space to toggle rectification
        show_rectify = not show_rectify
    elif key == 27:  # ESC key to exit
        break

video.release()
cv.destroyAllWindows()