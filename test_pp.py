import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
import json

mpl.use('macosx')
base_path = '/Users/aribakhan/Dropbox (MIT)/shared_Khan/final_data/OutputData/'
base_file = 'MyExpt_Trial_Cell_'

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

num_imgs = c2_df['ImageNumber'].nunique()

c2_df.set_index('ImageNumber', inplace=True)
c3_df.set_index('ImageNumber', inplace=True)
c4_df.set_index('ImageNumber', inplace=True)
c5_df.set_index('ImageNumber', inplace=True)
c6_df.set_index('ImageNumber', inplace=True)


def generate_random_colors(N):
    np.random.seed(5423)
    df = pd.Series(np.random.randint(10, 50, N), index=np.arange(1, N + 1))

    cmap = plt.cm.tab10
    colors = cmap(np.arange(len(df)) % cmap.N)

    return colors


def graph_clusters(center_coords, n, threshold):
    plt.figure(figsize=(5.511013215859031, 5.511013215859031))
    plt.scatter(c2_df['AreaShape_Center_X'].loc[n], c2_df['AreaShape_Center_Y'].loc[n], color='#FF00FF', s=12,
                label=c2_df.name)
    plt.scatter(c3_df['AreaShape_Center_X'].loc[n], c3_df['AreaShape_Center_Y'].loc[n], color='r', s=12,
                label=c3_df.name)
    plt.scatter(c4_df['AreaShape_Center_X'].loc[n], c4_df['AreaShape_Center_Y'].loc[n], color='y', s=12,
                label=c4_df.name)
    plt.scatter(c5_df['AreaShape_Center_X'].loc[n], c5_df['AreaShape_Center_Y'].loc[n], color='g', s=12,
                label=c5_df.name)
    plt.scatter(c6_df['AreaShape_Center_X'].loc[n], c6_df['AreaShape_Center_Y'].loc[n], color='b', s=12,
                label=c6_df.name)
    plt.title('Cell Location w/ Center Coordinates (IMG-{})'.format(n))
    plt.legend(loc='best')
    plt.xlabel('Horizontal Distance - X Coordinate (Pxls)')
    plt.ylabel('Vertical Distance - Y Coordinate (Pxls)')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    map_data = {}
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

    colors = generate_random_colors(len(fin_dict))
    for i in range(len(fin_dict)):
        plt.scatter(fin_keys[i][0], fin_keys[i][1], s=120, color=colors[i], alpha=0.3)
        plt.scatter(fin_dict_x[fin_keys[i]], fin_dict_y[fin_keys[i]], s=120, facecolors='none', edgecolors=colors[i])

    name = str(n) + '.png'
    plt.savefig('/Users/aribakhan/Dropbox (MIT)/shared_Khan/final_data/threshold65/' + name)

    # plt.show()


######### Determine and Remove Cells w/ Same Center ############
final_object_data = []
images_obj = {}
for n in range(1, num_imgs + 1):
    threshold = 65
    unique_cc = {}
    unique_xc = {}
    unique_yc = {}

    center_coords = [(e, f) for c in channel_names for e, f in
                     zip(c['AreaShape_Center_X'].loc[n], c['AreaShape_Center_Y'].loc[n])]
    images_obj[n] = center_coords
    center_coords_x = [e for c in channel_names for e in
                       c['AreaShape_Center_X'].loc[n]]

    dupl_cc = set([center_coords[j] for i in range(len(center_coords)) for j in range(i + 1, len(center_coords)) if
                   (math.hypot(center_coords[i][0] - center_coords[j][0],
                               center_coords[i][1] - center_coords[j][1]) <= threshold)])

    dupl_clusters = set([(center_coords[i][0], center_coords[j][0]) for i in range(len(center_coords))
                         for j in range(i + 1, len(center_coords)) if
                         (math.hypot(center_coords[i][0] - center_coords[j][0],
                                     center_coords[i][1] - center_coords[j][1])
                          <= threshold)])

    unique_cc[n] = [x for x in center_coords if x not in list(dupl_cc)]
    unique_xc[n] = [x[0] for x in center_coords if x not in list(dupl_cc)]
    unique_yc[n] = [x[1] for x in center_coords if x not in list(dupl_cc)]

    unique_cc = unique_cc[n]
    unique_xc = unique_xc[n]
    unique_yc = unique_yc[n]
    graph_clusters(center_coords, n, threshold)
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
                    channel_2[i] = c2_df.loc[
                        c2_df['AreaShape_Center_X'] == j, 'Intensity_MeanIntensity_phase_c2'].item()
                elif j in list(c3_df['AreaShape_Center_X']):
                    channel_3[i] = c3_df.loc[
                        c3_df['AreaShape_Center_X'] == j, 'Intensity_MeanIntensity_phase_c3'].item()
                elif j in list(c4_df['AreaShape_Center_X']):
                    channel_4[i] = c4_df.loc[
                        c4_df['AreaShape_Center_X'] == j, 'Intensity_MeanIntensity_phase_c4'].item()
                elif j in list(c5_df['AreaShape_Center_X']):
                    channel_5[i] = c5_df.loc[
                        c5_df['AreaShape_Center_X'] == j, 'Intensity_MeanIntensity_phase_c5'].item()
                elif j in list(c6_df['AreaShape_Center_X']):
                    channel_6[i] = c6_df.loc[
                        c6_df['AreaShape_Center_X'] == j, 'Intensity_MeanIntensity_phase_c6'].item()
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

    image_num = [n for _ in range(len(num_identifier))]
    num_identifier = [i + 1 for i in num_identifier]
    unique_objects_data = pd.DataFrame(
        np.column_stack([image_num, num_identifier, unique_xc, unique_yc, unique_maj, unique_min, unique_angle,
                         channel_2, channel_3, channel_4, channel_5, channel_6]),
        columns=['Image Number', 'Object Number', 'Center X', 'Center Y', 'Major Axis', 'Minor Axis', 'Angle',
                 'Channel 2', 'Channel 3', 'Channel 4', 'Channel 5', 'Channel 6'])
    final_object_data.append(unique_objects_data)
fin_unique_objects_data = pd.concat(final_object_data, ignore_index=True)

json_obj = json.dumps(images_obj, indent=4)
json_file = open("/Users/aribakhan/Dropbox (MIT)/shared_Khan/final_data/data.json", "w")
json_file.write(json_obj)
