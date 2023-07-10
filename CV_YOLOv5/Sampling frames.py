
import cv2
import os

cam = cv2.VideoCapture("james.mp4")

try:

    if not os.path.exists('dataj'):
        os.makedirs('dataj')

except OSError:
    print('Error: Creating directory of data')

# frame
currentframe = 0
frame_rate = 60
count=0
while(True):

    ret, frame = cam.read()

    if ret:
        currentframe += 1

        if (currentframe % frame_rate == 0):
            count+=1
            name = './dataj/frame' + str(count) + '.jpg'
            print('Creating...' + name)

            cv2.imwrite(name, frame)

    else:
        break

# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()
