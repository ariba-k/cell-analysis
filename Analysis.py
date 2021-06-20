from PreProcessing import unique_objects_data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.ticker as mtick

base_path = '/Users/aribakhan/Dropbox (MIT)/shared_Khan/histograms/'

# NORMALIZE DATA
for ind in unique_objects_data.index:
    unique_objects_data.at[ind, 'Channel 2 Norm'] = \
        (unique_objects_data['Channel 2'][ind] - unique_objects_data['Channel 2'].min()) / \
        (unique_objects_data['Channel 2'].max() - unique_objects_data['Channel 2'].min())
    unique_objects_data.at[ind, 'Channel 3 Norm'] = \
        (unique_objects_data['Channel 3'][ind] - unique_objects_data['Channel 3'].min()) / \
        (unique_objects_data['Channel 3'].max() - unique_objects_data['Channel 3'].min())
    unique_objects_data.at[ind, 'Channel 4 Norm'] = \
        (unique_objects_data['Channel 4'][ind] - unique_objects_data['Channel 4'].min()) \
        / (unique_objects_data['Channel 4'].max() - unique_objects_data['Channel 4'].min())
    unique_objects_data.at[ind, 'Channel 5 Norm'] = \
        (unique_objects_data['Channel 5'][ind] - unique_objects_data['Channel 5'].min()) / \
        (unique_objects_data['Channel 5'].max() - unique_objects_data['Channel 5'].min())
    unique_objects_data.at[ind, 'Channel 6 Norm'] = \
        (unique_objects_data['Channel 6'][ind] - unique_objects_data['Channel 6'].min()) / \
        (unique_objects_data['Channel 6'].max() - unique_objects_data['Channel 6'].min())
# move thresholding to before normalization
# threshold based on raw and then normalize

# print("Channel 2: % .3f" % unique_objects_data['Channel 2'].min())
# print(unique_objects_data['Channel 3'].min())
# print(unique_objects_data['Channel 4'].min())
# print(unique_objects_data['Channel 5'].min())
# print(unique_objects_data['Channel 6'].min())

# INTENSITY SUM - sum the normalized data
unique_objects_data.loc[:, 'Intensity Sum'] = unique_objects_data[unique_objects_data.columns[-5:]].sum(axis=1)

# WAVELENGTH - calculate wavelength emission of each channel
for ind in unique_objects_data.index:
    unique_objects_data.at[ind, 'C2 Wavelength'] = \
        (unique_objects_data['Channel 2 Norm'][ind] / unique_objects_data['Intensity Sum'][ind]) * 664
    unique_objects_data.at[ind, 'C3 Wavelength'] = \
        (unique_objects_data['Channel 3 Norm'][ind] / unique_objects_data['Intensity Sum'][ind]) * 610
    unique_objects_data.at[ind, 'C4 Wavelength'] = \
        (unique_objects_data['Channel 4 Norm'][ind] / unique_objects_data['Intensity Sum'][ind]) * 526
    unique_objects_data.at[ind, 'C5 Wavelength'] = \
        (unique_objects_data['Channel 5 Norm'][ind] / unique_objects_data['Intensity Sum'][ind]) * 517
    unique_objects_data.at[ind, 'C6 Wavelength'] = \
        (unique_objects_data['Channel 6 Norm'][ind] / unique_objects_data['Intensity Sum'][ind]) * 463

# WEIGHTED SUM - sum the wavelengths
unique_objects_data["Weighted Sum"] = unique_objects_data[unique_objects_data.columns[-5:]].sum(axis=1)
color = ['m', 'r', 'y', 'g', 'b']


# CHANNEL DISTRIBUTION - histogram graph
def generate_channel_dist():
    norm = []
    raw = []
    i = 0
    fig, ax = plt.subplots(6, 2, figsize=(8, 10), tight_layout=True)
    plt.tight_layout()
    raw_count = []
    raw_bins = []
    norm_count = []
    norm_bins = []

    raw_threshold = [0.01, 0.01, 0.01, 0.025, 0.005]
    for cr, cn in zip(unique_objects_data[unique_objects_data.columns[6:11]],
                      unique_objects_data[unique_objects_data.columns[11:16]]):
        # filter_unique_raw = unique_objects_data[unique_objects_data[cr] > raw_threshold[i]]
        filter_unique_raw = unique_objects_data.loc[unique_objects_data[cr] > raw_threshold[i]]
        # filter_unique_norm = unique_objects_data.loc[unique_objects_data[cn] > raw_threshold[i] /
        #                                               unique_objects_data[cr].max()]
        # print(math.ceil(math.sqrt(len(filter_unique_raw[cr].tolist()))))

        raw.append(filter_unique_raw[cr].tolist())
        counts_raw, bins_raw = np.histogram(np.asarray(filter_unique_raw[cr].tolist()))
        raw_count.append(counts_raw)
        raw_bins.append(bins_raw)


        # ax[i, 0].hist(filter_unique_raw[cr].tolist(), bins=15,
        #               color=color[unique_objects_data.columns.get_loc(cr) - 6], edgecolor='black', linewidth=1.2)
        if i>2:
            width_size =max(bins_raw)/18.5
        else:
            width_size = max(bins_raw) / 11.5

        ax[i, 0].bar(bins_raw[:-1], counts_raw,  width=width_size, color=color[unique_objects_data.columns.get_loc(cr) - 6], edgecolor='black')
        ax[i, 0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=len(filter_unique_raw[cr].tolist())))

        ax[i, 0].set_title(cr + ' Raw')
        ax[i, 0].set_xlabel('Raw Mean Intensity')
        ax[i, 0].set_yscale('log')

        raw_max = max(filter_unique_raw[cr].tolist())
        raw_min = min(filter_unique_raw[cr].tolist())

        filter_unique_norm = [(i - raw_min) / (raw_max - raw_min) for i in filter_unique_raw[cr].tolist()]

        norm.append(filter_unique_norm)
        counts_norm, bins_norm = np.histogram(np.asarray(filter_unique_norm))
        norm_count.append(counts_norm)
        norm_bins.append(bins_norm)
        # print(counts_norm)
        # print(bins_norm)
        ax[i, 1].set_title(cn)
        # print(len(filter_unique_norm[cn].tolist()))
        # math.ceil(math.sqrt(len(filter_unique_raw[cr].tolist())))
        ax[i, 1].bar(bins_norm[:-1], counts_norm,  width=0.1, color=color[unique_objects_data.columns.get_loc(cr) - 6], edgecolor='black')
        # ax[i, 1].hist(filter_unique_norm, bins=math.ceil(math.sqrt(len(filter_unique_raw[cr].tolist()))),
        #               color=color[unique_objects_data.columns.get_loc(cr) - 6],
        #               edgecolor='black', linewidth=1.2)
        ax[i, 1].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=len(filter_unique_norm)))
        ax[i, 1].set_xlabel('Norm Mean Intensity')
        ax[i, 1].set_xlim(min(bins_norm), max(bins_norm))
        # ax[i, 1].set_yscale('log')


        i += 1
    # print((filter_unique_raw != 0).astype(int).sum(axis=0))

    # math.ceil(math.sqrt(len(raw[i])))

    for i, c in enumerate(color):
        # print(len(raw))
        if i > 2:
            width_size = max(raw_bins[i]) / 11.5
        else:
            width_size = max(raw_bins[i]) / 11.5
        ax[5, 0].bar(raw_bins[i][:-1], raw_count[i], width=width_size, color=c, edgecolor='black')
        # ax[5, 0].hist(raw[i], bins=15, color=c, alpha=0.5, edgecolor='black', linewidth=1.2)

        # ax[5, 1].hist(norm[i], bins=15, color=c, alpha=0.5, edgecolor='black', linewidth=1.2)
        ax[5, 1].bar(norm_bins[i][:-1], norm_count[i], width=0.1, color=c, edgecolor='black')

    norm_count = 0
    for n in norm:
        norm_count += len(n)
    raw_count = 0
    for r in raw:
        raw_count += len(r)
    ax[5, 1].set_title("Norm Histograms Overlaid")
    ax[5, 1].set_xlabel('Norm Mean Intensity')
    ax[5, 1].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=norm_count))
    ax[5, 1].set_xlim(xmin=0)
    # ax[5, 1].set_yscale('log')
    ax[5, 0].set_title("Raw Histograms Overlaid")
    ax[5, 0].set_xlabel('Raw Mean Intensity')
    ax[5, 0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=raw_count))
    ax[5, 0].set_xlim(xmin=0)
    ax[5, 0].set_yscale('log')

    #plt.savefig(base_path + "channel_distribution/" + 'bar_dist' + '.png')
    #plt.show()

    raw_path = '/Users/aribakhan/Dropbox (MIT)/shared_Khan/data/raw/'
    norm_path = '/Users/aribakhan/Dropbox (MIT)/shared_Khan/data/norm/'

    c2_raw_df = pd.DataFrame(data=raw[0], columns=['C2 Raw'])
    c3_raw_df = pd.DataFrame(data=raw[1], columns=['C3 Raw'])
    c4_raw_df = pd.DataFrame(data=raw[2], columns=['C4 Raw'])
    c5_raw_df = pd.DataFrame(data=raw[3], columns=['C4 Raw'])
    c6_raw_df = pd.DataFrame(data=raw[4], columns=['C6 Raw'])

    c2_raw_df.to_csv(raw_path + 'c2_raw.csv')
    c3_raw_df.to_csv(raw_path + 'c3_raw.csv')
    c4_raw_df.to_csv(raw_path + 'c4_raw.csv')
    c5_raw_df.to_csv(raw_path + 'c5_raw.csv')
    c6_raw_df.to_csv(raw_path + 'c6_raw.csv')

    c2_norm_df = pd.DataFrame(data=norm[0], columns=['C2 Norm'])
    c3_norm_df = pd.DataFrame(data=norm[1], columns=['C3 Norm'])
    c4_norm_df = pd.DataFrame(data=norm[2], columns=['C4 Norm'])
    c5_norm_df = pd.DataFrame(data=norm[3], columns=['C4 Norm'])
    c6_norm_df = pd.DataFrame(data=norm[4], columns=['C6 Norm'])

    c2_norm_df.to_csv(norm_path + 'c2_norm.csv')
    c3_norm_df.to_csv(norm_path + 'c3_norm.csv')
    c4_norm_df.to_csv(norm_path + 'c4_norm.csv')
    c5_norm_df.to_csv(norm_path + 'c5_norm.csv')
    c6_norm_df.to_csv(norm_path + 'c6_norm.csv')




    return raw, norm


generate_channel_dist()


# CELL DISTRIBUTION - bar graph
def generate_cell_dist():
    ind = ['C2', 'C3', 'C4', 'C5', 'C6']
    for index, row in unique_objects_data[unique_objects_data.columns[11:16]].iterrows():
        object_name = str(int(unique_objects_data.iloc[index]['Object Number']) + 1)
        coord_name = '(' + str(round(unique_objects_data.iloc[index]['Center X'], 2)) + ',' + \
                     str(round(unique_objects_data.iloc[index]['Center Y'], 2)) + ')'
        title = object_name + ':' + coord_name
        plt.figure()
        row = [x / unique_objects_data['Intensity Sum'][index] for x in row.tolist()]
        bar = plt.bar(ind, row, 0.35)
        for i, c in enumerate(color):
            bar[i].set_color(c)
        plt.title(title)
        # plt.savefig(base_path + "cell_distribution/" + object_name + '.png')
        # plt.show()

# generate_cell_dist()
