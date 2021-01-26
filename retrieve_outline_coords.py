import cv2
import numpy as np
import matplotlib as plt
import json

# Read image; find contours with hierarchy
base_path = '/Users/aribakhan/Dropbox (MIT)/shared_Khan/cell_segmentation_data/mock data/'
image = cv2.imread(base_path + 'test_angle.png')
im_bw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

(thresh, im_bw) = cv2.threshold(im_bw, 128, 255, 0)
cv2.imwrite(base_path + 'test_angle_bw.jpg', im_bw)

contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
cv2.imwrite(base_path + 'test_angle_cntr.jpg', image)

sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

# for i in range(len(contours)):
#     print(contours[i])
object_coords = {}
for i, c in enumerate(sorted_contours):
    M = cv2.moments(c)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    cv2.putText(image, text=str(i + 1), org=(cx, cy),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255),
                thickness=1, lineType=cv2.LINE_AA)
    object_coords[str(i + 1)] = c.tolist()

cv2.imwrite(base_path + 'test_angle_cntr_num.jpg', image)

json_obj = json.dumps(object_coords, indent=2)

with open(base_path + "object_coords.json", "w") as f:
    f.write(json_obj)
