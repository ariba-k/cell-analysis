from PreProcessing import *
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import math

output_path = '/Users/aribakhan/Dropbox (MIT)/shared_Khan/images/'
input_path = '/Users/aribakhan/Dropbox (MIT)/shared_Khan/cell_segmentation_data/InputData/'

# img = plt.imread("/Users/aribakhan/PycharmProject/FluoroCellSegAnalysis /Combined_Cell_Overlay.jpg")
# # img2 = plt.imread("/Users/aribakhan/PycharmProject/FluoroCellSegAnalysis /Tile_05_c2m27_Overlay.jpg")
# # img3 = plt.imread("/Users/aribakhan/PycharmProject/FluoroCellSegAnalysis /Tile_05_c3m27_Overlay.jpg")
# # img4 = plt.imread("/Users/aribakhan/PycharmProject/FluoroCellSegAnalysis /Tile_05_c4m27_Overlay.jpg")
# # img5 = plt.imread("/Users/aribakhan/PycharmProject/FluoroCellSegAnalysis /Tile_05_c5m27_Overlay.jpg")
# # img6 = plt.imread("/Users/aribakhan/PycharmProject/FluoroCellSegAnalysis /Tile_05_c6m27_Overlay.jpg")

img = plt.imread("/Users/aribakhan/Dropbox (MIT)/shared_Khan/microscopy/Tile05_Ape1-6_Live_20x_EM/png/Tile_05_c2+3+4"
                 "+5+6m27.png")
img2 = plt.imread(input_path+'Channel 2/Tile_05_c2m27.png')
img3 = plt.imread(input_path+'Channel 3/Tile_05_c3m27.png')
img4 = plt.imread(input_path+'Channel 4/Tile_05_c4m27.png')
img5 = plt.imread(input_path+'Channel 5/Tile_05_c5m27.png')
img6 = plt.imread(input_path+'Channel 6/Tile_05_c6m27.png')

unique_angle = [-math.degrees(a + 2 * math.pi) if a < 0 else math.degrees(a) for a in unique_angle]
# c2_angle = [math.degrees(a+math.pi/2) if a < 0 else math.degrees(a) for a in c2_df['AreaShape_Orientation'].loc[1]]
# c2_angle = [-(np.arctan(a)*(180/math.pi)) if a < 0 else (np.arctan(a)*(180/math.pi)) for a in c2_df['AreaShape_Orientation'].loc[1]]
# c2_angle = [-math.degrees(np.arctan(a)) if a < 0 else math.degrees(np.arctan(a)) for a in c2_df['AreaShape_Orientation'].loc[1]]
# c2_angle = [-math.degrees(a + 2 * math.pi) if a < 0 else math.degrees(a) for a in c2_df['AreaShape_Orientation'].loc[1]]
c2_angle = [-math.degrees(a + 2 * math.pi) if a < 0 else math.degrees(a) for a in c2_df['AreaShape_Orientation'].loc[1]]
c3_angle = [-math.degrees(a + 2 * math.pi) if a < 0 else math.degrees(a) for a in c3_df['AreaShape_Orientation'].loc[1]]
c4_angle = [-math.degrees(a + 2 * math.pi) if a < 0 else math.degrees(a) for a in c4_df['AreaShape_Orientation'].loc[1]]
c5_angle = [-math.degrees(a + 2 * math.pi) if a < 0 else math.degrees(a) for a in c5_df['AreaShape_Orientation'].loc[1]]
c6_angle = [-math.degrees(a + 2 * math.pi) if a < 0 else math.degrees(a) for a in c6_df['AreaShape_Orientation'].loc[1]]

# print(c2_angle)
# print(c2_df['AreaShape_Orientation'].loc[1])

ells = [Ellipse((x_c, y_c), width=w, height=h, angle=a) for x_c, y_c, w, h, a in
        zip(unique_xc, unique_yc, unique_min, unique_maj, unique_angle)]
ells2 = [Ellipse((x_c, y_c), width=w, height=h, angle=a) for x_c, y_c, w, h, a in
         zip(c2_df['AreaShape_Center_X'].loc[1], c2_df['AreaShape_Center_Y'].loc[1],
             c2_df['AreaShape_MinorAxisLength'].loc[1], c2_df['AreaShape_MajorAxisLength'].loc[1],
             c2_angle)]
ells3 = [Ellipse((x_c, y_c), width=w, height=h, angle=a) for x_c, y_c, w, h, a in
         zip(c3_df['AreaShape_Center_X'].loc[1], c3_df['AreaShape_Center_Y'].loc[1],
             c3_df['AreaShape_MinorAxisLength'].loc[1], c3_df['AreaShape_MajorAxisLength'].loc[1],
             c3_angle)]
ells4 = [Ellipse((x_c, y_c), width=w, height=h, angle=a) for x_c, y_c, w, h, a in
         zip(c4_df['AreaShape_Center_X'].loc[1], c4_df['AreaShape_Center_Y'].loc[1],
             c4_df['AreaShape_MinorAxisLength'].loc[1], c4_df['AreaShape_MajorAxisLength'].loc[1],
             c4_angle)]
ells5 = [Ellipse((x_c, y_c), width=w, height=h, angle=a) for x_c, y_c, w, h, a in
         zip(c5_df['AreaShape_Center_X'].loc[1], c5_df['AreaShape_Center_Y'].loc[1],
             c5_df['AreaShape_MinorAxisLength'].loc[1], c5_df['AreaShape_MajorAxisLength'].loc[1],
             c5_angle)]
ells6 = [Ellipse((x_c, y_c), width=w, height=h, angle=a) for x_c, y_c, w, h, a in
         zip(c6_df['AreaShape_Center_X'].loc[1], c6_df['AreaShape_Center_Y'].loc[1],
             c6_df['AreaShape_MinorAxisLength'].loc[1], c6_df['AreaShape_MajorAxisLength'].loc[1],
             c6_angle)]

### COMBINED #####
# fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
# for e in ells:
#     ax.add_artist(e)
#     e.set_clip_box(ax.bbox)
#     e.set_alpha(0.6)
#     e.set_facecolor(np.random.rand(3))
#
# ax.set_xlim(0, 1024)
# ax.set_ylim(0, 1024)
plt.figure()
plt.title("Combined")
for x, y, n in zip(unique_xc, unique_yc, num_identifier):
    title = str(round(x, 2)) + "," + str(round(y, 2))
    plt.text(x - 7, y + 7, title, color='white', size=5)
plt.gca().invert_yaxis()
#ax.imshow(img)
plt.imshow(img)
plt.savefig(output_path + '/coordinates/actual/' + 'combined.png')

### C2 #####
# fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
# for e in ells2:
#     ax.add_artist(e)
#     e.set_clip_box(ax.bbox)
#     e.set_alpha(0.6)
#     e.set_facecolor(np.random.rand(3))
#
# ax.set_xlim(0, 1024)
# ax.set_ylim(0, 1024)
plt.figure()
plt.title("Channel 2")
for x, y, n in zip(c2_df['AreaShape_Center_X'].loc[1], c2_df['AreaShape_Center_Y'].loc[1],
                   c2_df['ObjectNumber'].loc[1]):
    title = str(round(x, 2)) + "," + str(round(y, 2))
    plt.text(x - 7, y + 7, title, color='white', size=5)

plt.gca().invert_yaxis()
# ax.imshow(img2)
plt.imshow(img2)
plt.savefig(output_path + '/coordinates/actual/' + 'c2.png')

### C3 #####
# fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
# for e in ells3:
#     ax.add_artist(e)
#     e.set_clip_box(ax.bbox)
#     e.set_alpha(0.6)
#     e.set_facecolor(np.random.rand(3))
#
# ax.set_xlim(0, 1024)
# ax.set_ylim(0, 1024)
plt.figure()
plt.title("Channel 3")
for x, y, n in zip(c3_df['AreaShape_Center_X'].loc[1], c3_df['AreaShape_Center_Y'].loc[1],
                   c3_df['ObjectNumber'].loc[1]):
    title = str(round(x, 2)) + "," + str(round(y, 2))
    plt.text(x - 7, y + 7, title, color='white', size=5)
plt.gca().invert_yaxis()
# ax.imshow(img3)
plt.imshow(img3)
plt.savefig(output_path + '/coordinates/actual/' + 'c3.png')

### C4 #####
# fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
# for e in ells4:
#     ax.add_artist(e)
#     e.set_clip_box(ax.bbox)
#     e.set_alpha(0.6)
#     e.set_facecolor(np.random.rand(3))
#
# ax.set_xlim(0, 1024)
# ax.set_ylim(0, 1024)
plt.figure()
plt.title("Channel 4")
for x, y, n in zip(c4_df['AreaShape_Center_X'].loc[1], c4_df['AreaShape_Center_Y'].loc[1],
                   c4_df['ObjectNumber'].loc[1]):
    title = str(round(x, 2)) + "," + str(round(y, 2))
    plt.text(x - 7, y + 7, title, color='white', size=5)
plt.gca().invert_yaxis()
# ax.imshow(img4)
plt.imshow(img4)
plt.savefig(output_path + 'coordinates/actual/' + 'c4.png')

### C5 #####
# fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
# for e in ells5:
#     ax.add_artist(e)
#     e.set_clip_box(ax.bbox)
#     e.set_alpha(0.6)
#     e.set_facecolor(np.random.rand(3))
#
# ax.set_xlim(0, 1024)
# ax.set_ylim(0, 1024)
plt.figure()
plt.title("Channel 5")
for x, y, n in zip(c5_df['AreaShape_Center_X'].loc[1], c5_df['AreaShape_Center_Y'].loc[1],
                   c5_df['ObjectNumber'].loc[1]):
    title = str(round(x, 2)) + "," + str(round(y, 2))
    plt.text(x - 7, y + 7, title, color='white', size=5)
plt.gca().invert_yaxis()
#ax.imshow(img5)
plt.imshow(img5)
plt.savefig(output_path + 'coordinates/actual/' + 'c5.png')

### C6 #####
# fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
# for e in ells6:
#     ax.add_artist(e)
#     e.set_clip_box(ax.bbox)
#     e.set_alpha(0.6)
#     e.set_facecolor(np.random.rand(3))
#
# ax.set_xlim(0, 1024)
# ax.set_ylim(0, 1024)
plt.figure()
plt.title("Channel 6")
for x, y, n in zip(c6_df['AreaShape_Center_X'].loc[1], c6_df['AreaShape_Center_Y'].loc[1],
                   c6_df['ObjectNumber'].loc[1]):
    title = str(round(x, 2)) + "," + str(round(y, 2))
    plt.text(x - 7, y + 7, title, color='white', size=5)
plt.gca().invert_yaxis()
#ax.imshow(img6)
plt.imshow(img6)
plt.savefig(output_path + 'coordinates/actual/' + 'c6.png')

plt.show()
