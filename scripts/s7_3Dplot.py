import flopy
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmcrameri as cm
import matplotlib.ticker as mticker
import pyvista as pv
import sys
sys.path.append(os.path.join("..","particle_track"))
from particle_track import pollock_v2
from s3_particle_track_TOC_model import vectorized_p

## Model 1: facies model
# loading datasets from the previous models:
facies = np.load("facies_v7.npy")
#facies = facies.astype(np.int16)
# loading the model:
unique = np.unique(facies, return_counts=True)
unique[1]/np.sum(unique[1])
ws = "ammer_v7"
## loading the model:
ammer_model = flopy.mf6.MFSimulation.load(sim_ws=ws)
gwf = ammer_model.get_model("ammer_V07")
nrow = gwf.modelgrid.nrow
nlay = gwf.modelgrid.nlay
ncol = gwf.modelgrid.ncol

# particle tracking center cross section outflow points
cellcenters = gwf.modelgrid.xyzcellcenters
# get the ibound array
ibound = gwf.dis.idomain.array
np.random.seed(69)
random_indexes_row = np.random.choice(np.arange(0, nrow), 100)
randow_indexes_lay = np.random.choice(np.arange(0, nlay), 100)
z = cellcenters[2][randow_indexes_lay,random_indexes_row,-1]
ibound_center = ibound[randow_indexes_lay,random_indexes_row,-1]
z_center = z[ibound_center == 1]
x = cellcenters[0][random_indexes_row,-1]
x = x[ibound_center == 1]
y = cellcenters[1][random_indexes_row,-1]
y = y[ibound_center == 1]

layer = randow_indexes_lay[ibound_center == 1]
row = random_indexes_row[ibound_center == 1]
col = np.repeat([ncol-1], len(z_center))
points = np.stack([x,y,z_center, layer, row, col]).T

pt = pollock_v2(
    gwfmodel=gwf,
    model_directory=ws,
    particles_starting_location=points,
    porosity=vectorized_p(facies),
    mode="backwards",
    processes=8,)

def process_points(pt_results, facies, reactivity):
    vertices = []
    lines = []
    k = 0
    for j in range(np.max(pt_results[:, 0]).astype(np.int32) + 1):
        results = pt_results[pt_results[:, 0] == j, 1:]
        results = np.flipud(results)
        results = results[1:,:]
        size = results.shape[0]
        ts = results[:, 3]
        points = np.column_stack((results[:, 0], results[:, 1], results[:, 2]))
        line = np.arange(results.shape[0]) + k
        k += size
        line_arr = np.array([size])
        line_arr = np.concatenate([line_arr, line])
        vertices.append(points)
        lines.append(line_arr)
    # lines = np.vstack(lines)
    # vertices = np.vstack(vertices)
    return lines, vertices


lines, vertices = process_points(pt, facies, np.load(os.path.join(ws,"f_.npy")))
vertices = np.vstack(vertices)
lines = np.hstack(lines)
mesh = pv.PolyData(vertices, lines=lines)
tube = mesh.tube(radius=2)


delc, delr, delz = gwf.modelgrid.delc, gwf.modelgrid.delr, gwf.modelgrid.delz
xcorn = np.concatenate([[0], np.cumsum(delr)])
ycorn = np.concatenate([[0], np.cumsum(delc)])
zcorn = np.concatenate([[0], np.cumsum(delz[:,0,0])])
pvgrid = pv.RectilinearGrid(xcorn, ycorn, zcorn)
pvgrid = pv.RectilinearGrid.cast_to_structured_grid(pvgrid)
# pvgrid.plot(show_edges=True)
# add facies array as cell data
# because of how the data is aligned I can just flip the array.
pvgrid.cell_data["facies"] = np.flip(facies.flatten())
# instantaneous reactable concentration of nitrate
colors_full = {2:"#cec47f",4:"#c9c186",6:"#b0b468",7:"#c8b23d",8:"#323021", 9:"#5a1919", 10:"#3c5579", 11:"#FCF2A8", 12:"#b7b7b7"}
keys = (np.array(list(colors_full.keys()))-2)/10
pvfacies = np.flip(facies.ravel())
mapping = np.linspace(2, 12, 256)
pyvista_colors = np.empty((256,4))
pyvista_colors[mapping > 11] = np.concatenate([mpl.colors.to_rgb("#b7b7b7"),[1.]])
pyvista_colors[mapping <= 11.1] = np.concatenate([mpl.colors.to_rgb("#FCF2A8"),[1.]])
pyvista_colors[mapping <= 10.1] = np.concatenate([mpl.colors.to_rgb("#3c5579"),[1.]])
pyvista_colors[mapping <= 9.1] = np.concatenate([mpl.colors.to_rgb("#5a1919"),[1.]])
pyvista_colors[mapping <= 8.1] = np.concatenate([mpl.colors.to_rgb("#323021"),[1.]])
pyvista_colors[mapping <= 7.1] = np.concatenate([mpl.colors.to_rgb("#c8b23d"),[1.]])
pyvista_colors[mapping <= 6.1] = np.concatenate([mpl.colors.to_rgb("#b0b468"),[1.]])
pyvista_colors[mapping <= 4.1] = np.concatenate([mpl.colors.to_rgb("#c9c186"),[1.]])
pyvista_colors[mapping <= 3.1] = np.concatenate([mpl.colors.to_rgb("#cec47f"),[1.]])

# Create a colormap
my_cmap = mpl.colors.ListedColormap(pyvista_colors)
cmap = mpl.colors.ListedColormap(list(colors_full.values()))
bounds = [2,4,6,7,8,9,10,11,12,13]
# Create a normalization object
z_scale = 7
pvgrid = pvgrid.scale([1,1,z_scale])
horizontal_slice = pvgrid.slice_orthogonal(x=0, y=0, z=max(zcorn)*z_scale/3+0.2)
vertical_slice = pvgrid.slice_orthogonal(x=0, y=max(ycorn)/2, z=0)
cross_slice = pvgrid.slice_orthogonal(x=max(xcorn)/2, y=0, z=0)


vertices = np.vstack(vertices)
vertices[:, 2] = vertices[:, 2]*z_scale
lines = np.hstack(lines)
mesh = pv.PolyData(vertices, lines=lines)
tube = mesh.tube(radius=2)

p = pv.Plotter()
# lt = pv.LookupTable(values = pyvista_colors, value_range=[2,12])
# lt.below_range_color = pv.Color('grey', opacity=0.5)
# lt.above_range_color = pv.Color('grey', opacity=0.5)
#label = pv.Label("Cross section", position = [450, 300, 12])
# rect = pv.Rectangle(points = [[0,max(ycorn)/2,0],[max(xcorn),max(ycorn)/2,0],[max(xcorn),max(ycorn)/2,max(zcorn)]])
p.add_mesh(horizontal_slice, scalars="facies", show_edges=False, cmap=my_cmap, clim=[2,12], show_scalar_bar=False, opacity=0.75)
p.add_mesh(vertical_slice, scalars="facies", show_edges=False, cmap=my_cmap, clim=[2,12], show_scalar_bar=False, opacity=0.75)
# p.add_mesh(horizontal_slice2, scalars="facies", show_edges=False, cmap=my_cmap, clim=[2,12], show_scalar_bar=False, opacity=0.75)
p.add_mesh(cross_slice, scalars="facies", show_edges=False, cmap=my_cmap, clim=[2,12], show_scalar_bar=False, opacity=0.75)
#p.add_actor(label)
#show axis
p.add_axes()
p.add_mesh(tube, color="lightblue", opacity=1)
#p.set_scale(1, 1, 1)
# light = pv.Light()
# light.set_direction_angle(5, 30)
# p.add_light(light)
def callback(x):
    print(x) # not really relevant here
    print(f'camera position: {p.camera.position}')
    print(f'camera az,rol,elev: {p.camera.azimuth},{p.camera.roll},\
    {p.camera.elevation}')
    print(f'camera vi   ew angle, focal point: {p.camera.view_angle,p.camera.focal_point}')
# now set the camera parameters in the code
p.track_click_position(callback)
p.show()


p = pv.Plotter(lighting="three lights")
p.add_mesh(horizontal_slice, scalars="facies", show_edges=False, cmap=my_cmap,
    clim=[2,12], show_scalar_bar=False, opacity=0.6, smooth_shading=True)
p.add_mesh(vertical_slice, scalars="facies", show_edges=False, cmap=my_cmap,
    clim=[2,12], show_scalar_bar=False, opacity=0.6, smooth_shading=True)
p.add_mesh(cross_slice, scalars="facies", show_edges=False, cmap=my_cmap,
    clim=[2,12], show_scalar_bar=False, opacity=0.6, smooth_shading=True)
#p.add_actor(label)
#show axis
#p.add_axes()
p.add_mesh(tube, color="lightsteelblue", opacity=1,label="flowlines")
# legend = ["flowlines", "lightsteelblue", pv.Tube]
p.add_legend()

light = pv.Light(position=(-44.93262274069701, -562.1735877035144, 891.4477317331155), intensity=0.2)
#light.set_direction_angle(0, 75)
p.add_light(light)
p.camera.position = (-44.93262274069701, -562.1735877035144, 891.4477317331155)
p.camera.azimuth = 0
p.camera.roll = 35.74549756268309
p.camera.elevation = 0
p.enable_anti_aliasing('msaa', multi_samples=64)
p.camera.view_angle = 30
p.camera.focal_point = (443.5874591386224, 298.77606179629225, -137.37247739174023)
p.save_graphic("3D_plot.pdf")
p.save_graphic("3D_plot.svg")
p.save_graphic("3D_plot.eps")

def callback_print(x):
    p.save_graphic("3D_plot_click.svg", raster=False)
    p.screenshot("3D_plot_click.png", transparent_background=True, scale = 10)
# now set the camera parameters in the code
p.track_click_position(callback_print)
p.show(screenshot='3D_plot.png')