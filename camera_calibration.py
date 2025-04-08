import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)

fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('chessboard.avi', fourcc, 20.0, (960, 540))
recording = False

print("Press SPACE to toggle Record/Pause, ESC to exit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    if recording:
        out.write(frame)
        cv.circle(frame, (50, 50), 20, (0, 0, 255), -1)
    
    cv.imshow('Video Recorder', frame)
    key = cv.waitKey(1) & 0xFF
    
    if key == 32:
        recording = not recording
        print(f"Mode: {'Recording' if recording else 'Paused'}")
    elif key == 27: 
        break

cap.release()
out.release()
cv.destroyAllWindows()

def select_img_from_video(video_file, board_pattern, select_all=False, wait_msec=10):
    video = cv.VideoCapture(video_file)
    img_select = []
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        complete, _ = cv.findChessboardCorners(gray, board_pattern)
        if complete or select_all:
            img_select.append(frame)
        if wait_msec > 0:
            cv.imshow('Frame', frame)
            if cv.waitKey(wait_msec) & 0xFF == 27:
                break
    video.release()
    cv.destroyAllWindows()
    return img_select

# Function to calibrate camera
def calib_camera_from_chessboard(images, board_pattern, board_cellsize, K=None, dist_coeff=None, calib_flags=None):
    img_points = []
    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        complete, pts = cv.findChessboardCorners(gray, board_pattern)
        if complete:
            img_points.append(pts)
    assert len(img_points) > 0, 'No complete chessboard points found!'
    obj_pts = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    obj_points = [np.array(obj_pts, dtype=np.float32) * board_cellsize] * len(img_points)
    return cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], K, dist_coeff, flags=calib_flags)

board_pattern = (9, 6) 
board_cellsize = 0.025
images = select_img_from_video('chessboard.avi', board_pattern)
if images:
    ret, K, dist_coeff, rvecs, tvecs = calib_camera_from_chessboard(images, board_pattern, board_cellsize)
    print(f"Number of applied images: {len(images)}")
    print(f"RMS error: {ret}")
    print("Camera Matrix (K):")
    print(K)
    print("Distortion Coefficients:")
    print(dist_coeff)
else:
    print("No suitable frames found.")