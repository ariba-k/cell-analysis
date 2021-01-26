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
#
# color = ['m', 'r', 'y', 'g', 'b']
# norm = []
# raw = []
# # plt.figure()
# # for cr, cn in zip(unique_objects_data[unique_objects_data.columns[6:11]],
# #                   unique_objects_data[unique_objects_data.columns[12:17]]):
# #     filter_unique_norm = unique_objects_data[unique_objects_data[cn] > 0.1]
# #     plt.subplot(2, 1, 1)
# #     plt.hist(unique_objects_data[cr].tolist(), color=color[unique_objects_data.columns.get_loc(cr) - 6],
# #              edgecolor='black', linewidth=1.2)
# #     raw.append(unique_objects_data[cr].tolist())
# #     plt.title(cr + ' Raw')
# #     plt.xlabel('Raw Mean Intensity')
# #     plt.tight_layout()
# #     plt.subplot(2, 1, 2)
# #     norm.append(filter_unique_norm[cn].tolist())
# #     plt.hist(filter_unique_norm[cn].tolist(), color=color[unique_objects_data.columns.get_loc(cr) - 6],
# #              edgecolor='black', linewidth=1.2)
# #     plt.title(cn)
# #     plt.xlabel('Norm Mean Intensity')
# #     plt.tight_layout()
# #     plt.savefig(base_path + "channel_distribution/" + cr + '_comb' + '.png')
# #     ##plt.show()
#
# fig, ax = plt.subplots()
# for i, c in enumerate(color):
#     plt.title("Norm Histograms Overlaid")
#     plt.xlabel('Norm Mean Intensity')
#     ax.hist(norm[i], color=c, alpha=0.5, edgecolor='black', linewidth=1.2)
# plt.savefig(base_path + "channel_distribution/" + 'norm_overlay' + '.png')
#
# fig2, ax2 = plt.subplots()
# for i, c in enumerate(color):
#     plt.title("Raw Histograms Overlaid")
#     plt.xlabel('Raw Mean Intensity')
#     ax2.hist(raw[i], color=c, alpha=0.5, edgecolor='black', linewidth=1.2)
# plt.savefig(base_path + "channel_distribution/" + 'raw_overlay' + '.png')
# #
# #
#
#
# # CELL DISTRIBUTION - bar graph
# ind = ['C2', 'C3', 'C4', 'C5', 'C6']
# values = []
# for index, row in unique_objects_data[unique_objects_data.columns[11:16]].iterrows():
#     object_name = str(int(unique_objects_data.iloc[index]['Object Number']) + 1)
#     coord_name = '(' + str(round(unique_objects_data.iloc[index]['Center X'], 2)) + ',' + \
#                  str(round(unique_objects_data.iloc[index]['Center Y'], 2)) + ')'
#     title = object_name + ':' + coord_name
#     plt.figure()
#     row = [x / unique_objects_data['Intensity Sum'][index] for x in row.tolist()]
#     bar = plt.bar(ind, row, 0.35)
#     for i, c in enumerate(color):
#         bar[i].set_color(c)
#     plt.title(title)
#     plt.savefig(base_path + "cell_distribution/" + object_name + '.png')

# #
#
#
# #
# X = np.arange(115)
# c2 = [i[0] for i in values]
# c3 = [i[1] for i in values]
# c4 = [i[2] for i in values]
# c5 = [i[3] for i in values]
# c6 = [i[4] for i in values]
#
# vals = [c2, c3, c4, c5, c6]
# width = 0.8
# plt.figure(figsize=(15, 5))
# n = len(vals)
# _X = np.arange(len(X))
# for i in range(n):
#     plt.bar(_X - width / 2. + i / float(n) * width, vals[i],
#             width=width / float(n), color=color[i], align="center")
#
# plt.autoscale(tight=True)
# plt.tight_layout()
# plt.xticks(X[::3], X[::3])
# plt.ylim(0, 8)
# plt.ylabel("Intensity")
# plt.xlabel("Cell Number")
# plt.savefig(base_path + "cell_distribution/" + 'grouped' + '.png')

# #plt.show()
