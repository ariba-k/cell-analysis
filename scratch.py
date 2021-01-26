import matplotlib.pyplot as plt
from Analysis import unique_objects_data

base_path = '/Users/aribakhan/Dropbox (MIT)/shared_Khan/histograms/'
color = ['m', 'r', 'y', 'g', 'b']
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
    ax[i, 1].hist(filter_unique_norm[cn].tolist(), bins=30, color=color[unique_objects_data.columns.get_loc(cr) - 6],
                  edgecolor='black', linewidth=1.2)
    ax[i, 1].set_xlabel('Norm Mean Intensity')
    ax[i, 1].set_xlim(xmin=0)
    ax[i, 1].set_yscale('log')

    i += 1

for i, c in enumerate(color):
    ax[5, 0].hist(raw[i], bins=30, color=c, alpha=0.5, edgecolor='black', linewidth=1.2)
    ax[5, 1].hist(norm[i], bins=30, color=c, alpha=0.5, edgecolor='black', linewidth=1.2)

ax[5, 1].set_title("Norm Histograms Overlaid")
ax[5, 1].set_xlabel('Norm Mean Intensity')
ax[5, 1].set_xlim(xmin=0)
ax[5, 1].set_yscale('log')
ax[5, 0].set_title("Raw Histograms Overlaid")
ax[5, 0].set_xlabel('Raw Mean Intensity')
ax[5, 0].set_xlim(xmin=0)
ax[5, 0].set_yscale('log')

plt.savefig(base_path + "channel_distribution/" + 'w_filter_comb' + '.png')


plt.tight_layout()
plt.show()

print(raw)
print(norm)
