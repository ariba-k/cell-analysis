from PreProcessing import unique_objects_data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

base_path = '/Users/aribakhan/Dropbox (MIT)/shared_Khan/histograms/'

# NORMALIZE DATA
for ind in unique_objects_data.index:
    unique_objects_data.at[ind, 'Channel 2 Norm'] = \
        unique_objects_data['Channel 2'][ind] / unique_objects_data['Channel 2'].max()
    unique_objects_data.at[ind, 'Channel 3 Norm'] = \
        unique_objects_data['Channel 3'][ind] / unique_objects_data['Channel 3'].max()
    unique_objects_data.at[ind, 'Channel 4 Norm'] = \
        unique_objects_data['Channel 4'][ind] / unique_objects_data['Channel 4'].max()
    unique_objects_data.at[ind, 'Channel 5 Norm'] = \
        unique_objects_data['Channel 5'][ind] / unique_objects_data['Channel 5'].max()
    unique_objects_data.at[ind, 'Channel 6 Norm'] = \
        unique_objects_data['Channel 6'][ind] / unique_objects_data['Channel 6'].max()

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
    raw_threshold = [0.01, 0.01, 0.01, 0.025, 0.005]
    for cr, cn in zip(unique_objects_data[unique_objects_data.columns[6:11]],
                      unique_objects_data[unique_objects_data.columns[11:16]]):
        # filter_unique_raw = unique_objects_data[unique_objects_data[cr] > raw_threshold[i]]
        filter_unique_raw = unique_objects_data.loc[unique_objects_data[cr] > raw_threshold[i]]
        filter_unique_norm = unique_objects_data.loc[unique_objects_data[cn] > raw_threshold[i] /
                                                     unique_objects_data[cr].max()]

        ax[i, 0].hist(filter_unique_raw[cr].tolist(), bins=30, color=color[unique_objects_data.columns.get_loc(cr) - 6],
                      edgecolor='black', linewidth=1.2)
        raw.append(filter_unique_raw[cr].tolist())
        ax[i, 0].set_title(cr + ' Raw')
        ax[i, 0].set_xlabel('Raw Mean Intensity')
        ax[i, 0].set_xlim(xmin=0)
        ax[i, 0].set_yscale('log')

        norm.append(filter_unique_norm[cn].tolist())
        ax[i, 1].set_title(cn)
        ax[i, 1].hist(filter_unique_norm[cn].tolist(), bins=30,
                      color=color[unique_objects_data.columns.get_loc(cr) - 6],
                      edgecolor='black', linewidth=1.2)
        ax[i, 1].set_xlabel('Norm Mean Intensity')
        ax[i, 1].set_xlim(xmin=0)
        #ax[i, 1].set_yscale('log')

        i += 1

    for i, c in enumerate(color):
        ax[5, 0].hist(raw[i], bins=30, color=c, alpha=0.5, edgecolor='black', linewidth=1.2)
        ax[5, 1].hist(norm[i], bins=30, color=c, alpha=0.5, edgecolor='black', linewidth=1.2)

    ax[5, 1].set_title("Norm Histograms Overlaid")
    ax[5, 1].set_xlabel('Norm Mean Intensity')
    ax[5, 1].set_xlim(xmin=0)
    #ax[5, 1].set_yscale('log')
    ax[5, 0].set_title("Raw Histograms Overlaid")
    ax[5, 0].set_xlabel('Raw Mean Intensity')
    ax[5, 0].set_xlim(xmin=0)
    ax[5, 0].set_yscale('log')

    plt.savefig(base_path + "channel_distribution/" + 'w_filter_comb_revised' + '.png')
    plt.show()


#generate_channel_dist()


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
        plt.savefig(base_path + "cell_distribution/" + object_name + '.png')
        #plt.show()

#generate_cell_dist()