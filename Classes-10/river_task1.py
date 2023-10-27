import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from RiverModel_vol2 import RiverModel
# 3D plot
import plotly.graph_objects as go
import plotly.io as pio

from matplotlib import cm
from matplotlib.colors import LightSource

# ------------- SETTINGS ------------- #
Nx = 300  # 300
Ny = 200  # 200
const_I = 1.0
delta_h = 10
beta = 0.05
const_r = 200

RiverClass = RiverModel(Nx=Nx, Ny=Ny, const_I=const_I, delta_h=delta_h, beta=beta, const_r=const_r)

# --- create plot: evolve the height
# n_droplet_test = 17000
#
# RiverClass.droplets_till_end_vol2(n_droplet_test)
# river_h0 = RiverClass.return_grid_h()
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title(f"height map after: {n_droplet_test + 1} droplet", fontsize=15, loc='left')
# cax1 = ax.imshow(river_h0, interpolation='nearest', cmap='jet')
# plt.colorbar(cax1)
# plt.savefig(f'./height-map-droplet-{n_droplet_test}.png')
# # plt.show()
# plt.close(fig)

# --- create plot: evolve the height (with evalance)
# n_droplet_test = 14000
#
# RiverClass.droplets_till_end_vol2(n_droplet_test)
# river_h0 = RiverClass.return_grid_h()
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title(f"height map after: {n_droplet_test + 1} droplet", fontsize=15, loc='left')
# cax1 = ax.imshow(river_h0, interpolation='nearest', cmap='jet')
# plt.colorbar(cax1)
# plt.savefig(f'./height-map-droplet-one-avalanches-{n_droplet_test}.png')
# # plt.show()
# plt.close(fig)

# --- create plot: evolve the height (with evalance)
# n_droplet_test = 20000
#
# RiverClass.droplets_till_end(n_droplet_test)
# river_h0 = RiverClass.return_grid_h()
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title(f"height map after: {n_droplet_test + 1} droplet", fontsize=15, loc='left')
# cax1 = ax.imshow(river_h0, interpolation='nearest', cmap='jet')
# plt.colorbar(cax1)
# plt.savefig(f'./height-map-droplet-avalanches-{n_droplet_test}.png')
# # plt.show()
# plt.close(fig)


# --- PLOT RIVER
n_droplet_test = 15100
part_to_river = 1000
simple_threshold = 200

# when_river = [12000, 14000, 14400, 14800, 15200]
# when_river = [12000, 14000, 14400, 14800, 15200]
# when_river = [14000, 14400, 14800, 14900, 15200]
when_river = [15050]



for i in range(0, n_droplet_test):
    print("Which droplet?:", i)
    RiverClass.one_droplet_till_end()

    if i in when_river:
        RiverClass.path_of_river()

river_h0 = RiverClass.return_river_grid(threshold=simple_threshold)

# Create a colormap with two colors: white and blue
cmap = plt.cm.colors.ListedColormap(['white', 'blue'])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title(f"River after: {n_droplet_test} droplet", fontsize=15, loc='left')
cax1 = ax.imshow(river_h0, interpolation='nearest', cmap=cmap)
plt.colorbar(cax1)
plt.savefig(f'./River-map-{n_droplet_test}.png')
# plt.show()
plt.close(fig)

# --- PLOT 3D
topography = RiverClass.return_grid_h()
fig = go.Figure(data=[go.Surface(z=topography)])
# fig.update_layout(title='Mt Bruno Elevation', autosize=False,
#                   width=500, height=500,
#                   margin=dict(l=65, r=50, b=65, t=90))
fig.update_layout(title='Area topography', autosize=True)
# Save the plot as an image
# pio.write_image(fig, f'topography-plot-{n_droplet_test}.png')
fig.show()

# --- PLOT 3D
# z = topography
# nrows, ncols = z.shape
# x = np.linspace(0, Nx, ncols)
# y = np.linspace(0, Ny, nrows)
# x, y = np.meshgrid(x, y)
# # Set up plot
# river_map = RiverClass.return_river_grid(threshold=simple_threshold)
# fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
#
# ls = LightSource(270, 45)
# # To use a custom hillshading mode, override the built-in shading and pass
# # in the rgb colors of the shaded surface calculated from "shade".
# # cm.gist_earth
# rgb = ls.shade(river_map, cmap='jet')
# surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
#                        linewidth=0, antialiased=False, shade=False)
# # plt.colorbar(rgb)
# plt.savefig(f'./3D-River-and-topography-map-{n_droplet_test}.png')
# plt.show()

