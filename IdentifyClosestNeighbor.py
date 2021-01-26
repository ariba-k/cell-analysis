from PreProcessing import *


######### Determine Coordinates - Ellipse ############
# reference: http://quickcalcbasic.com/ellipse%20line%20intersection.pdf

# Parametric Equation of an Ellipse:
# x = h cos β cos α − v sin β sin α + e
# y = v sin β cos α + h cos β sin α + f

# h = major radius, v = minor radius
# β = parameter from 0 to 360 degrees, α = angle of rotation (x axis and major axis)
# e = center x coordinate, f = center y coordinate

object_info = {}

threshold = 5
# Trial with Channel 2 Objects
for img_num, obj_num, e, f, h, v, alpha in zip(c2_df['ImageNumber'], c2_df['ObjectNumber'], c2_df['AreaShape_Center_X'],
                                               c2_df['AreaShape_Center_Y'],
                                               c2_df['AreaShape_MajorAxisLength'] / 2,
                                               c2_df['AreaShape_MinorAxisLength'] / 2,
                                               c2_df['AreaShape_Orientation']):
    circum_points = [(e, f)]
    for beta in range(360):
        x, y = (h * math.cos(beta) * math.cos(alpha) - v * math.sin(beta) * math.sin(alpha) + e,
                v * math.sin(beta) * math.cos(alpha) + h * math.cos(beta) * math.sin(alpha) + f)
        circum_points.append((x, y))
    object_info[img_num, obj_num] = circum_points

# Determine Objects Touching
# 1) Center Points Overlap x<x+threshold or x>x-threshold
# 2) Circum Points x<x+threshold or x>x-threshold

x_c = [coord[0][0] for coord in object_info.values()]
y_c = [coord[0][1] for coord in object_info.values()]

same_xc = []
same_yc = []

for i in range(len(x_c)):
    for j in range(i + 1, len(x_c)):
        if (x_c[i] - threshold) <= x_c[j] <= (x_c[i] + threshold):
            same_xc.append([x_c[i], x_c[j]])


for i in range(len(y_c)):
    for j in range(i + 1, len(y_c)):
        if (y_c[i] - threshold) <= y_c[j] <= (y_c[i] + threshold):
            same_yc.append([y_c[i], y_c[j]])

x = []
y = []
for obj in object_info.values():
    x_dup = []
    for coord in obj:
        x_dup.append(coord[0])
    x.append(x_dup)

for obj in object_info.values():
    y_dup = []
    for coord in obj:
        y_dup.append(coord[1])
    y.append(y_dup)

same_x = []
for obj in x:
    for i in range(len(obj)):
        for j in range(i + 1, len(obj)):
            if (obj[i] - threshold) <= obj[j] <= (obj[i] + threshold):
                same_x.append([obj[i], obj[j]])

same_y = []
for obj in y:
    for i in range(len(obj)):
        for j in range(i + 1, len(obj)):
            if (obj[i] - threshold) <= obj[j] <= (obj[i] + threshold):
                same_y.append([obj[i], obj[j]])


