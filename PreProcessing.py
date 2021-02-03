import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import csv
from matplotlib.patches import Ellipse

base_path = '/Users/aribakhan/Dropbox (MIT)/shared_Khan/cell_segmentation_data/OutputDataFluorescent/'
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

eccentricity_df.to_csv(base_path + 'eccentricity.csv', index=False, header=True)

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

plt.figure(figsize=(5.511013215859031, 5.511013215859031))
plt.scatter(c2_df['AreaShape_Center_X'].loc[2], c2_df['AreaShape_Center_Y'].loc[2], color='#FF00FF', s=12,
            label=c2_df.name)
plt.scatter(c3_df['AreaShape_Center_X'].loc[2], c3_df['AreaShape_Center_Y'].loc[2], color='r', s=12, label=c3_df.name)
plt.scatter(c4_df['AreaShape_Center_X'].loc[2], c4_df['AreaShape_Center_Y'].loc[2], color='y', s=12, label=c4_df.name)
plt.scatter(c5_df['AreaShape_Center_X'].loc[2], c5_df['AreaShape_Center_Y'].loc[2], color='g', s=12, label=c5_df.name)
plt.scatter(c6_df['AreaShape_Center_X'].loc[2], c6_df['AreaShape_Center_Y'].loc[2], color='b', s=12, label=c6_df.name)
plt.title('Cell Location w/ Center Coordinates (IMG-28)')
plt.legend(loc='best')
plt.xlabel('Horizontal Distance - X Coordinate (Pxls)')
plt.ylabel('Vertical Distance - Y Coordinate (Pxls)')
plt.gca().invert_yaxis()
plt.tight_layout()
# plt.show()

######### Determine and Remove Cells w/ Same Center ############
#plt.figure()
threshold = 10
# center_coords = [(e, f) for c in channel_names for e, f in
#                  zip(c['AreaShape_Center_X'].loc[1], c['AreaShape_Center_Y'].loc[1])]
# center_coords_x = [e for c in channel_names for e in
#                    c['AreaShape_Center_X'].loc[1]]
unique_cc = []
unique_xc = []
unique_yc = []
for n in range(1, num_imgs+1):
    center_coords = [(e, f) for c in channel_names for e, f in
                     zip(c['AreaShape_Center_X'].loc[n], c['AreaShape_Center_Y'].loc[n])]
    center_coords_x = [e for c in channel_names for e in
                       c['AreaShape_Center_X'].loc[n]]
    #print(str(n) + ":" + str(center_coords))
    #print(str(n) + ":" + str(len(center_coords)))
    dupl_cc = set([center_coords[j] for i in range(len(center_coords)) for j in range(i + 1, len(center_coords)) if
                   (math.hypot(center_coords[i][0] - center_coords[j][0],
                               center_coords[i][1] - center_coords[j][1]) <= threshold)])

    dupl_clusters = set([(center_coords[i][0], center_coords[j][0]) for i in range(len(center_coords))
                         for j in range(i + 1, len(center_coords)) if (math.hypot(center_coords[i][0] - center_coords[j][0],
                                                                                  center_coords[i][1] - center_coords[j][1])
                                                                       <= threshold)])

    unique_cc_inc = [x for x in center_coords if x not in list(dupl_cc)]
    unique_cc.append(unique_cc_inc)
    unique_xc_inc = [c[0] for c in unique_cc_inc]
    unique_xc.append(unique_xc_inc)
    unique_yc_inc = [c[1] for c in unique_cc_inc]
    unique_yc.append(unique_yc_inc)

unique_cc = [c_c for n in unique_cc for c_c in n]
unique_xc = [x_c for n in unique_xc for x_c in n]
unique_yc = [y_c for n in unique_yc for y_c in n]


#plt.scatter(unique_xc, unique_yc, s=12)

# plt.show()

######### Reorganize Cell Objects ############

# Format: Number Identifier,
#         Center X, Center Y,
#         Major Axis, Minor Axis,
#         Angle,
#         Channel 2, Channel 3, Channel 4, Channel 5, Channel 6

num_identifier = [num for num in range(0, len(unique_cc))]
unique_xc = [c[0] for c in unique_cc]
unique_yc = [c[1] for c in unique_cc]
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

unique_objects_data = pd.DataFrame(
    np.column_stack([num_identifier, unique_xc, unique_yc, unique_maj, unique_min, unique_angle,
                     channel_2, channel_3, channel_4, channel_5, channel_6]),
    columns=['Object Number', 'Center X', 'Center Y', 'Major Axis', 'Minor Axis', 'Angle',
             'Channel 2', 'Channel 3', 'Channel 4', 'Channel 5', 'Channel 6'])

# print(unique_objects_data[unique_objects_data.columns[-5:]])
