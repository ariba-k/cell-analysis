import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
import numpy as np
import math
import pandas as pd

# TESTING ANGLES

# 1) Identify Object numbers in each channel where angles do not seem accurate
# 2) Plot Ellipses of those objects in a separate graph and do a side by side comparison w/ original image
# 3) Create mock ellipse cell objects with known angles
# 4) Run through CellProfiler 

# MOCK DATA
angles = [8, 30, 10, 50, 78]
center_x = [200, 550, 867, 343, 612]
center_y = [340, 945, 277, 533, 712]
width = [43, 52, 63, 40, 65]
height = [24, 65, 69, 70, 65]
obj_num = [n for n in range(1, len(angles) + 1)]

ells_m = [Ellipse((x_c, y_c), width=w, height=h, angle=a) for x_c, y_c, w, h, a in
          zip(center_x, center_y, width, height, angles)]

fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
for e in ells_m:
    ax.add_artist(e)
    e.set_clip_box(ax.bbox)
    e.set_alpha(0.6)
    e.set_facecolor('white')

for x, y, n in zip(center_x, center_y, obj_num):
    plt.text(x - 5, y + 5, str(n), color='red', size=8)

ax.set_xlim(0, 1024)
ax.set_ylim(0, 1024)
fig.patch.set_facecolor('black')
ax.axis('off')
plt.gca().invert_yaxis()

# SEGMENTED DATA
e_df = pd.read_csv('/Users/aribakhan/Dropbox (MIT)/shared_Khan/cell_segmentation_data/mock '
                   'data/MyExpt_irregular_cell_obj.csv')
img = plt.imread("/Users/aribakhan/Dropbox (MIT)/shared_Khan/cell_segmentation_data/mock "
                 "data/test_angle_irregular_overlay.jpeg")


#e_angle = [math.degrees(a+360) if a < 0 else math.degrees(a) for a in e_df['AreaShape_Orientation']]
#e_angle = [-math.degrees(a % (2 * math.pi)) if a < 0 else math.degrees(a) for a in e_df['AreaShape_Orientation']]
#e_angle = [math.degrees(a+math.pi/2) if a < 0 else math.degrees(a) for a in e_df['AreaShape_Orientation']]
#e_angle = [-(np.arctan(a)*(180/math.pi)) if a < 0 else (np.arctan(a)*(180/math.pi)) for a in e_df['AreaShape_Orientation']]
#e_angle = [-math.degrees(np.arctan(a)) if a < 0 else math.degrees(np.arctan(a)) for a in e_df['AreaShape_Orientation']]
e_angle = [-math.degrees(a + 2 * math.pi) if a < 0 else math.degrees(a + 2 * math.pi) for a in e_df['AreaShape_Orientation']]
print(e_angle)
ells = [Ellipse((x_c, y_c), width=w, height=h, angle=a) for x_c, y_c, w, h, a in
        zip(e_df['AreaShape_Center_X'], e_df['AreaShape_Center_Y'],
            e_df['AreaShape_MinorAxisLength'], e_df['AreaShape_MajorAxisLength'],
            e_angle)]

fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
for e in ells:
    ax.add_artist(e)
    e.set_clip_box(ax.bbox)
    e.set_alpha(0.6)
    e.set_facecolor(np.random.rand(3))

plt.title("Mock Data Irregular\n-math.degrees(a + 2 * math.pi)\nmath.degrees(a + 2 * math.pi)")
for x, y, n in zip(e_df['AreaShape_Center_X'], e_df['AreaShape_Center_Y'], e_df['ObjectNumber']):
    plt.text(x - 5, y + 5, str(n), color='white', size=8)
plt.gca().invert_yaxis()
ax.imshow(img)

plt.show()
