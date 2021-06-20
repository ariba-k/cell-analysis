import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib as mpl
from matplotlib import colors as mcolors
import copy

mpl.use('macosx')
import itertools
import csv
from matplotlib.patches import Ellipse

from collections import defaultdict

# base_path = '/Users/aribakhan/Dropbox (MIT)/shared_Khan/final_data/OutputData/'
base_path = '/Users/aribakhan/Dropbox (MIT)/shared_Khan/cell_segmentation_data/OutputDataFluorescentOLD/'

# base_file = 'MyExpt_Trial_Cell_'
base_file = 'Measurements_Cell_'

# Create data frames of each channel

c2_df = pd.read_csv(base_path + base_file + 'C2.csv')
c2_df.name = 'C2'

c3_df = pd.read_csv(base_path + base_file + 'C3.csv')
c3_df.name = 'C3'

c4_df = pd.read_csv(base_path + base_file + 'C4.csv')
c4_df.name = 'C4'

c5_df = pd.read_csv(base_path + base_file + 'C5.csv')
c5_df.name = 'C5'

c6_df = pd.read_csv(base_path + base_file + 'C6.csv')
c6_df.name = 'C6'

channel_names = [c2_df, c3_df, c4_df, c5_df, c6_df]

total_df = pd.concat(channel_names)
total_df.name = 'TOTAL'

######### Determine Shape - Eccentricity ############
num_imgs = c2_df['ImageNumber'].nunique()


def create_eccen_df(num_imgs):
    # Simplify
    eccentricity_list = []
    for c in channel_names:
        eccentricity_list.append({c.name + '_Image' + str(i + 1):
                                      c.query('ImageNumber==' + str(i + 1))['AreaShape_Eccentricity'] for i in
                                  range(num_imgs)})

    epsilon = 0.3
    eccent_count_list = []

    for channel in eccentricity_list:
        total_count = 0
        circular_count = 0
        elliptical_count = 0
        for image in channel:
            total = channel[image][channel[image] < 1].count()
            circular = channel[image][channel[image] <= epsilon].count()
            elliptical = channel[image][channel[image] >= 1 - epsilon].count()

            total_count += total
            circular_count += circular
            elliptical_count += elliptical

        eccent_count_list.append((total_count, circular_count, elliptical_count))

    channel_eccent_count = {}
    for i in range(len(eccentricity_list)):
        channel_eccent_count[channel_names[i].name] = eccent_count_list[i]

    eccentricity_df = pd.DataFrame({'Channel #': [c.name for c in channel_names],
                                    'Total': [i[0] for i in channel_eccent_count.values()],
                                    'Circular': [i[1] for i in channel_eccent_count.values()],
                                    'Other': [abs(i[0] - i[1] - i[2]) for i in channel_eccent_count.values()],
                                    'Elliptical': [i[2] for i in channel_eccent_count.values()],
                                    'Circular (%)': [i[1] / i[0] * 100 for i in
                                                     channel_eccent_count.values()],
                                    'Other (%)': [abs(i[0] - i[1] - i[2]) / i[0] * 100 for i in
                                                  channel_eccent_count.values()],
                                    'Elliptical (%)': [i[2] / i[0] * 100 for i in
                                                       channel_eccent_count.values()]
                                    })

    # eccentricity_df.to_csv(base_path + 'eccentricity.csv', index=False, header=True)

    return eccentricity_df


# eccentricity_df = create_eccen_df(num_imgs)

######### Create Histogram Graph - Eccentricity ############

def create_eccen_histo():
    fig, axs = plt.subplots(2, 3)

    # Channel 2
    axs[0, 0].hist(c2_df['AreaShape_Eccentricity'], color='#FF00FF')
    axs[0, 0].set_title(c2_df.name)

    # Channel 3
    axs[0, 1].hist(c3_df['AreaShape_Eccentricity'], color='r')
    axs[0, 1].set_title(c3_df.name)

    # Channel 4
    axs[0, 2].hist(c4_df['AreaShape_Eccentricity'], color='y')
    axs[0, 2].set_title(c4_df.name)

    # Channel 5
    axs[1, 0].hist(c5_df['AreaShape_Eccentricity'], color='g')
    axs[1, 0].set_title(c5_df.name)

    # Channel 6
    axs[1, 1].hist(c6_df['AreaShape_Eccentricity'], color='b')
    axs[1, 1].set_title(c6_df.name)

    # Total
    axs[1, 2].hist(total_df['AreaShape_Eccentricity'], color='k')
    axs[1, 2].set_title(total_df.name)

    plt.tight_layout()


# print(c2_df['AreaShape_Center_X'].sub(415.93).abs().idxmin())
# print(c2_df.loc[26, ['AreaShape_Center_X', 'AreaShape_Center_Y', 'Intensity_MeanIntensity_phase_c2']])
# print()
# print(c3_df['AreaShape_Center_X'].sub(415.93).abs().idxmin())
# print(c3_df.loc[80, ['AreaShape_Center_X', 'AreaShape_Center_Y', 'Intensity_MeanIntensity_phase_c3']])
# print()
# print(c4_df['AreaShape_Center_X'].sub(415.93).abs().idxmin())
# print(c4_df.loc[40, ['AreaShape_Center_X', 'AreaShape_Center_Y', 'Intensity_MeanIntensity_phase_c4']])
# print()
# print(c5_df['AreaShape_Center_X'].sub(415.93).abs().idxmin())
# print(c5_df.loc[34, ['AreaShape_Center_X', 'AreaShape_Center_Y', 'Intensity_MeanIntensity_phase_c5']])
# print()
# print(c6_df['AreaShape_Center_X'].sub(415.93).abs().idxmin())
# print(c6_df.loc[5, ['AreaShape_Center_X', 'AreaShape_Center_Y', 'Intensity_MeanIntensity_phase_c6']])


######### PLOTTING CELLS AS POINTS  ############

plt.figure(figsize=(5.511013215859031, 5.511013215859031))

c2_df.set_index('ImageNumber', inplace=True)
c3_df.set_index('ImageNumber', inplace=True)
c4_df.set_index('ImageNumber', inplace=True)
c5_df.set_index('ImageNumber', inplace=True)
c6_df.set_index('ImageNumber', inplace=True)

plt.scatter(c2_df['AreaShape_Center_X'].loc[1], c2_df['AreaShape_Center_Y'].loc[1], color='#FF00FF', s=12,
            label=c2_df.name)
plt.scatter(c3_df['AreaShape_Center_X'].loc[1], c3_df['AreaShape_Center_Y'].loc[1], color='r', s=12, label=c3_df.name)
plt.scatter(c4_df['AreaShape_Center_X'].loc[1], c4_df['AreaShape_Center_Y'].loc[1], color='y', s=12, label=c4_df.name)
plt.scatter(c5_df['AreaShape_Center_X'].loc[1], c5_df['AreaShape_Center_Y'].loc[1], color='g', s=12, label=c5_df.name)
plt.scatter(c6_df['AreaShape_Center_X'].loc[1], c6_df['AreaShape_Center_Y'].loc[1], color='b', s=12, label=c6_df.name)
plt.title('Cell Location w/ Center Coordinates (IMG-27)')
plt.legend(loc='best')
plt.xlabel('Horizontal Distance - X Coordinate (Pxls)')
plt.ylabel('Vertical Distance - Y Coordinate (Pxls)')
plt.gca().invert_yaxis()
plt.tight_layout()

######### Determine and Remove Cells w/ Same Center ############
# plt.figure()
threshold = 120
# center_coords = [(e, f) for c in channel_names for e, f in
#                  zip(c['AreaShape_Center_X'].loc[1], c['AreaShape_Center_Y'].loc[1])]
# center_coords_x = [e for c in channel_names for e in
#                    c['AreaShape_Center_X'].loc[1]]
unique_cc = {}
unique_xc = {}
unique_yc = {}

# try:
#     for n in range(1, num_imgs + 1):
#         center_coords = [(e, f) for c in channel_names for e, f in
#                          zip(c['AreaShape_Center_X'].loc[n], c['AreaShape_Center_Y'].loc[n])]
#         # center_coords = [(e, f) for c in channel_names for e, f in
#         #                  zip(c['AreaShape_Center_X'].loc[n], c['AreaShape_Center_Y'].loc[n])]
#         center_coords_x = [e for c in channel_names for e in
#                            c['AreaShape_Center_X'].loc[n]]
#         # print(str(n) + ":" + str(center_coords))
#         # print(str(n) + ":" + str(len(center_coords)))
#         dupl_cc = set([center_coords[j] for i in range(len(center_coords)) for j in range(i + 1, len(center_coords)) if
#                        (math.hypot(center_coords[i][0] - center_coords[j][0],
#                                    center_coords[i][1] - center_coords[j][1]) <= threshold)])
#
#         dupl_clusters = set([(center_coords[i][0], center_coords[j][0]) for i in range(len(center_coords))
#                              for j in range(i + 1, len(center_coords)) if
#                              (math.hypot(center_coords[i][0] - center_coords[j][0],
#                                          center_coords[i][1] - center_coords[j][1])
#                               <= threshold)])
#
#         # comparison of first element against all other elements
#         # and so forth until each element is compared against each other
#
#         unique_cc[n] = [x for x in center_coords if x not in list(dupl_cc)]
#         unique_xc[n] = [x[0] for x in center_coords if x not in list(dupl_cc)]
#         unique_yc[n] = [x[1] for x in center_coords if x not in list(dupl_cc)]
#
#
# except KeyError:
#     pass
map_data = {}
n = 1
center_coords = [(e, f) for c in channel_names for e, f in
                 zip(c['AreaShape_Center_X'].loc[n], c['AreaShape_Center_Y'].loc[n])]
center_coords_x = [e for c in channel_names for e in
                   c['AreaShape_Center_X'].loc[n]]
dupl_cc = set([center_coords[j] for i in range(len(center_coords)) for j in range(i + 1, len(center_coords)) if
               (math.hypot(center_coords[i][0] - center_coords[j][0],
                           center_coords[i][1] - center_coords[j][1]) <= threshold)])


for i in range(len(center_coords)):
    point = set()
    for j in range(i + 1, len(center_coords)):
        if math.hypot(center_coords[i][0] - center_coords[j][0],
                      center_coords[i][1] - center_coords[j][1]) <= threshold:
            point.add(center_coords[j])
    map_data[center_coords[i]] = point

keys = list(map_data.keys())
map_data_c = copy.deepcopy(map_data)
del_keys = set()
for k in keys:
    for k2, v in map_data.items():
        if k in v:
            del_keys.add(k)
            map_data_c[k2].update(map_data[k])

for k in map_data:
    if k in del_keys:
        del map_data_c[k]

fin_dict = {k: set(v) for k, v in map_data_c.items()}
count = 0
for v in fin_dict.values():
    count += len(v)

empty_keys = set()
for k in keys:
    if not map_data[k]:
        empty_keys.add(k)

# print(del_keys)
# print(len(del_keys))
# print(len(map_data))
# print(empty_keys)
# print(len(empty_keys))


dupl_clusters = set([(center_coords[i][0], center_coords[j][0]) for i in range(len(center_coords))
                     for j in range(i + 1, len(center_coords)) if
                     (math.hypot(center_coords[i][0] - center_coords[j][0],
                                 center_coords[i][1] - center_coords[j][1])
                      <= threshold)])

# comparison of first element against all other elements
# and so forth until each element is compared against each other
dupl_xc = [i[0] for i in dupl_cc]
dupl_yc = [i[1] for i in dupl_cc]
unique_cc[n] = [x for x in center_coords if x not in list(dupl_cc)]
unique_xc[n] = [x[0] for x in center_coords if x not in list(dupl_cc)]
unique_yc[n] = [x[1] for x in center_coords if x not in list(dupl_cc)]

# unique_cc --> uaed to be nested list
# unique_cc (2nd) --> flattens list

# INDEX 1 is Image 27
unique_cc = unique_cc[1]
unique_xc = unique_xc[1]
unique_yc = unique_yc[1]

# plt.scatter(unique_xc, unique_yc, s=12)

# plt.show()

######### Reorganize Cell Objects ############

# Format: Number Identifier,
#         Center X, Center Y,
#         Major Axis, Minor Axis,
#         Angle,
#         Channel 2, Channel 3, Channel 4, Channel 5, Channel 6

# identifies each unique cell as an obj
num_identifier = [num for num in range(0, len(unique_cc))]
unique_maj = [total_df.loc[total_df['AreaShape_Center_X'] == x_c, 'AreaShape_MajorAxisLength'].item()
              for x_c in unique_xc]
unique_min = [total_df.loc[total_df['AreaShape_Center_X'] == x_c, 'AreaShape_MinorAxisLength'].item()
              for x_c in unique_xc]
unique_angle = [total_df.loc[total_df['AreaShape_Center_X'] == x_c, 'AreaShape_Orientation'].item()
                for x_c in unique_xc]


def find_intersection(duplicates):
    out = {}
    for elem in duplicates:
        try:
            out[elem[0]].extend(elem[1:])
        except KeyError:
            out[elem[0]] = list(elem)
    return [tuple(values) for values in out.values()]


clusters = unique_xc[::]

channel_2 = [0] * len(unique_xc)
channel_3 = [0] * len(unique_xc)
channel_4 = [0] * len(unique_xc)
channel_5 = [0] * len(unique_xc)
channel_6 = [0] * len(unique_xc)

for x in find_intersection(dupl_clusters):
    for i, e in enumerate(clusters):
        if x[0] == e:
            clusters[i] = x

for i in num_identifier:
    if isinstance(clusters[i], tuple):
        for j in clusters[i]:
            if j in list(c2_df['AreaShape_Center_X']):
                channel_2[i] = c2_df.loc[c2_df['AreaShape_Center_X'] == j, 'Intensity_MeanIntensity_phase_c2'].item()
            elif j in list(c3_df['AreaShape_Center_X']):
                channel_3[i] = c3_df.loc[c3_df['AreaShape_Center_X'] == j, 'Intensity_MeanIntensity_phase_c3'].item()
            elif j in list(c4_df['AreaShape_Center_X']):
                channel_4[i] = c4_df.loc[c4_df['AreaShape_Center_X'] == j, 'Intensity_MeanIntensity_phase_c4'].item()
            elif j in list(c5_df['AreaShape_Center_X']):
                channel_5[i] = c5_df.loc[c5_df['AreaShape_Center_X'] == j, 'Intensity_MeanIntensity_phase_c5'].item()
            elif j in list(c6_df['AreaShape_Center_X']):
                channel_6[i] = c6_df.loc[c6_df['AreaShape_Center_X'] == j, 'Intensity_MeanIntensity_phase_c6'].item()
    else:
        if clusters[i] in list(c2_df['AreaShape_Center_X']):
            channel_2[i] = c2_df.loc[c2_df['AreaShape_Center_X'] == clusters[i],
                                     'Intensity_MeanIntensity_phase_c2'].item()
        elif clusters[i] in list(c3_df['AreaShape_Center_X']):
            channel_3[i] = c3_df.loc[c3_df['AreaShape_Center_X'] == clusters[i],
                                     'Intensity_MeanIntensity_phase_c3'].item()
        elif clusters[i] in list(c4_df['AreaShape_Center_X']):
            channel_4[i] = c4_df.loc[c4_df['AreaShape_Center_X'] == clusters[i],
                                     'Intensity_MeanIntensity_phase_c4'].item()
        elif clusters[i] in list(c5_df['AreaShape_Center_X']):
            channel_5[i] = c5_df.loc[c5_df['AreaShape_Center_X'] == clusters[i],
                                     'Intensity_MeanIntensity_phase_c5'].item()
        elif clusters[i] in list(c6_df['AreaShape_Center_X']):
            channel_6[i] = c6_df.loc[c6_df['AreaShape_Center_X'] == clusters[i],
                                     'Intensity_MeanIntensity_phase_c6'].item()

np.random.seed(5423)
N = 15
df = pd.Series(np.random.randint(10, 50, N), index=np.arange(1, N + 1))

cmap = plt.cm.tab10
colors = cmap(np.arange(len(df)) % cmap.N)

# fin_x = []
# fin_y = []
# for X, Y in zip(unique_xc, unique_yc):
#     dup_x = []
#     dup_y = []
#     for x, y in zip(dupl_xc, dupl_yc):
#         if math.hypot(X - x, Y - y) <= threshold:
#             dup_x.append(x)
#             dup_y.append(y)
#     if dup_x:
#         fin_x.append(dup_x)
#     if dup_y:
#         fin_y.append(dup_y)
fin_keys = list(fin_dict.keys())

fin_dict_x = {}
fin_dict_y = {}
for k, v in fin_dict.items():
    x = []
    y = []
    for i in v:
        x.append(i[0])
        y.append(i[1])
    fin_dict_x[k] = x
    fin_dict_y[k] = y

for i in range(len(fin_dict)):
    plt.scatter(fin_keys[i][0], fin_keys[i][1], s=120, facecolors='none', edgecolors=colors[i])
    plt.scatter(fin_dict_x[fin_keys[i]], fin_dict_y[fin_keys[i]], s=120, color=colors[i])
# plt.scatter(dupl_xc, dupl_yc, s=120, facecolors='none', edgecolors='aqua')

plt.show()
unique_objects_data = pd.DataFrame(
    np.column_stack([num_identifier, unique_xc, unique_yc, unique_maj, unique_min, unique_angle,
                     channel_2, channel_3, channel_4, channel_5, channel_6]),
    columns=['Object Number', 'Center X', 'Center Y', 'Major Axis', 'Minor Axis', 'Angle',
             'Channel 2', 'Channel 3', 'Channel 4', 'Channel 5', 'Channel 6'])
# print(unique_objects_data[unique_objects_data.columns[-5:]])

