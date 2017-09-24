from __future__ import division

###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import rgb2hex
from matplotlib.patches import Polygon
import matplotlib as mpl
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pickle

FS = 20
TICK_SIZE = 12

def plot_bar(df, col):
    """
    Create bar plot for input column and crime type
    """
    # ct = pd.crosstab(df[col], df['Primary Type'])
    # ct['Total'] = ct.sum(axis=1)
    # ct.T.plot.bar()
    fig, ax = plt.subplots(figsize=(8, 6))
    if col == 'Total':
        df['Primary Type'].value_counts().plot.bar(ax=ax)
    else:
        try:
            pd.crosstab(df['Primary Type'], df[col]).plot.bar(ax=ax)
        except ValueError:
            print 'Column not found in data frame.'
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=FS)
        plt.xlabel(col)
    plt.ylabel('Counts', fontsize=FS)
    plt.xlabel('Crime Type', fontsize=FS)
    plt.xticks(rotation=45, fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE)
    plt.tight_layout()
    plt.show()


def plot_time(df):
    """
    Create plot for crime counts as a function of time, resampled by month
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    df['Primary Type'].resample('M').count().plot(ax=ax)
    plt.ylabel('Counts', fontsize=FS)
    plt.xlabel('Time', fontsize=FS)
    plt.xticks(fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE)
    plt.title('Monthly crime counts in 2015 and 2016', fontsize=FS)
    plt.tight_layout()
    plt.show()


def plot_heatmap(df, col):
    """
    Create heatmap plot for input column and crime type
    """
    try:
        ct = pd.crosstab(df[col], df['Primary Type'])
    except ValueError:
        print 'Column not found in data frame.'
    # if normalize:
    #     scaler = MinMaxScaler()
    #     ct = scaler.fit_transform(ct)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(ct, ax=ax)
    if col == 'Weekday':
        plt.yticks(np.arange(7),
                   ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                   rotation=45)
    plt.xticks(rotation=45, fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE)
    plt.xlabel('Crime Type', fontsize=FS)
    plt.ylabel(col, fontsize=FS)
    plt.tight_layout()
    plt.show()


def plot_map_contour(df, crime_type, nbin=50):
    """
    Create crime distribution on map for input crime type or all crimes with 'Total' as input
    """
    # Location for Chicago city
    urcornerlat = 42.03
    urcornerlong = -87.51
    llcornerlat = 41.64
    llcornerlong = -87.95

    fig, ax = plt.subplots(figsize=(12, 8))
    m = Basemap(projection='merc', resolution='c',
                llcrnrlat=llcornerlat, urcrnrlat=urcornerlat,
                llcrnrlon=llcornerlong, urcrnrlon=urcornerlong, ax=ax)
    m.readshapefile('shapefiles/Community Areas/geo_export',
                    'communities')

    # Get the contour
    if crime_type == 'Total':
        new_df = df[['Latitude', 'Longitude']]
    else:
        try:
            new_df = df.loc[df['Primary Type'] == crime_type, ['Latitude', 'Longitude']]
        except ValueError:
            print 'Crime type not found in the data frame.'

    H, xedge, yedge = np.histogram2d(new_df['Latitude'],
                                     new_df['Longitude'],
                                     range=[[llcornerlat, urcornerlat],
                                            [llcornerlong, urcornerlong]],
                                     bins=nbin)
    xedge = (xedge[:nbin] + xedge[1:nbin + 1]) * 0.5
    yedge = (yedge[:nbin] + yedge[1:nbin + 1]) * 0.5
    x, y = np.meshgrid(yedge, xedge)
    cs = m.contourf(x, y, H, ax=ax, latlon=True, cmap='Reds')
    m.colorbar(cs, location='right', pad="5%")
    m.drawmapboundary()
    plt.show()


def plot_map_community(df, crime_type):
    """
    Produce the crimes in each community on map for certain crime type or all crimes with 'Total'
    """
    if crime_type == 'Total':
        counts = df['Community Area'].value_counts()
    else:
        try:
            counts = df.loc[df['Primary Type'] == crime_type, 'Community Area'].value_counts()
        except ValueError:
            print 'Crime type not found in the data frame.'

    # Min-max-standardize the count of crimes in each community
    vmin = 0
    vmax = max(counts)
    counts = (counts - vmin) / float(vmax - vmin)

    # Create Chicago map
    fig, ax = plt.subplots(figsize=(12, 8))
    urcornerlat = 42.03
    urcornerlong = -87.51
    llcornerlat = 41.64
    llcornerlong = -87.95
    m = Basemap(projection='merc', resolution='c',
                llcrnrlat=llcornerlat, urcrnrlat=urcornerlat,
                llcrnrlon=llcornerlong, urcrnrlon=urcornerlong, ax=ax)

    # Read in shapefiles for community area
    m.readshapefile('shapefiles/Community Areas/geo_export',
                    'communities')

    # For each community, save the standardized crime count as color
    # Save the community name for future use
    colors = []
    names = []
    # community_name = {}
    for shapedict in m.communities_info:
        community_id = int(shapedict['area_numbe'])
        # community_name[community_id] = shapedict['community']
        if community_id in counts:
            colors.append(counts[community_id])
        else:
            colors.append(0)
        names.append(shapedict['community'])
    # pickle.dump(community_name, open('input/community_name', 'w'))

    # Plot the colored polygon in each community, annotate the community name if standardized count is larger than 0.5.
    cmap = plt.cm.Reds
    for idx, seg in enumerate(m.communities):
        color = rgb2hex(cmap(colors[idx])[:3])
        poly = Polygon(seg, facecolor=color, edgecolor=color)
        ax.add_patch(poly)
        center = np.mean(seg, axis=0)
        if colors[idx] > 0.5:
            plt.text(center[0], center[1], names[idx], color='blue', fontsize=TICK_SIZE)

    ax, _ = mpl.colorbar.make_axes(plt.gca())
    mpl.colorbar.ColorbarBase(ax=ax, cmap=cmap, norm=mpl.colors.Normalize(vmin=0, vmax=1))
    m.drawmapboundary()
    plt.show()


def biplot(df, col, community=None):
    """
    Produce a biplot that shows a scatterplot of the reduced
    data and the projections of the original features.
    """
    try:
        ct = pd.crosstab(df[col], df['Primary Type'])
    except ValueError:
        print 'Column not found in data frame.'
    fig, ax = plt.subplots(figsize=(8, 6))
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(ct)
    feature_vectors = pca.components_.T

    # scatterplot of the reduced data
    ax.scatter(x=reduced_data[:, 0], y=reduced_data[:, 1],
                  facecolors='b', edgecolors='b', s=70, alpha=0.5)

    # we use scaling factors to make the arrows easier to see
    arrow_size, text_pos, displacement = 2000, 2500, 300
    if col == 'Community Area':
        arrow_size *= 1.5
        text_pos *= 1.5
        for i, (p1, p2) in enumerate(reduced_data):
            if abs(p1) > 4000:
                ax.text(p1 - displacement, p2,
                           community[ct.index[i]],
                           color='green', ha='center', va='center', fontsize=TICK_SIZE)
            elif abs(p2) > 4000:
                ax.text(p1, p2 + displacement,
                           community[ct.index[i]],
                           color='green', ha='center', va='center', fontsize=TICK_SIZE)
    elif col == 'Hour' or col == 'Month':
        if col == 'Month':
            arrow_size *= 0.5
            text_pos *= 0.5
        for i, (p1, p2) in enumerate(reduced_data):
            ax.text(p1 - displacement, p2, ct.index[i],
                       color='green', ha='center', va='center', fontsize=TICK_SIZE)

    # projections of the original features
    for i, v in enumerate(feature_vectors):
        ax.arrow(0, 0, arrow_size * v[0], arrow_size * v[1],
                    head_width=0.2, head_length=0.2, linewidth=2, color='red')
        if (v[0] ** 2 + v[1] ** 2) > 0.2:
            ax.text(v[0] * text_pos, v[1] * text_pos, ct.columns[i], color='black',
                       ha='center', va='center', fontsize=TICK_SIZE)

    ax.set_xlabel("Dimension 1", fontsize=FS)
    ax.set_ylabel("Dimension 2", fontsize=FS)
    ax.set_title("PC plane with original " + col + " feature projections.", fontsize=FS)
    plt.xticks(fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE)
    plt.tight_layout()
    plt.show()
