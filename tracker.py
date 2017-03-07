import cv2
import numpy as np
import sys

from filters import combined_threshold

from moviepy.editor import VideoFileClip

def tracker(img):
    return combined_threshold(img)

if __name__ == "__main__":
    # clip1 = VideoFileClip("lizard_1.mp4")
    # video_clip = clip1.fl_image(tracker)
    # video_clip.write_videofile("lizard_1_tracked.mp4", audio=False)
    cap = cv2.VideoCapture('lizard_short.mp4')

    while (cap.isOpened()):
        ret, frame = cap.read()

        result= tracker(frame)
        cv2.imshow('img', result)

        while True:
            k = cv2.waitKey(25)
            if k == 27:
                sys.exit(0)
            elif k == ord('n'):
                break