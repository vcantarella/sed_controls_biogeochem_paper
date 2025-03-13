import flopy
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmcrameri as cm
import matplotlib.ticker as mticker
import particle_track
import pyvista as pv
import sys
sys.path.append(os.path.join("..","particle_track"))
from particle_track import pollock_v2
from s3_particle_track_TOC_model import vectorized_p



hatches = {11:"o",2:".",4:"",6:"..",7:"..",8:"",9:"",12:"oo", 10:""}
labels = {11:"T1",2:"T2",4:"T4",6:"T6",7:"T7",8:"T8",9:"T9",12:"G1",10:"C1"}
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
z = cellcenters[2][:,:,:]
z_center = z[:,int(nrow/2),-1]
ibound_center = ibound[:,int(nrow/2),-1]
z_center = z_center[ibound_center == 1]
x = np.repeat([899.9], len(z_center))
y = np.repeat([300], len(z_center))

layer = np.arange(0, nlay, 1)[ibound_center == 1]
row = np.repeat([int(nrow/2)], len(z_center))
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
horizontal_slice = pvgrid.slice_orthogonal(x=0, y=0, z=max(zcorn)/3+0.2)
vertical_slice = pvgrid.slice_orthogonal(x=0, y=max(ycorn)/2, z=0)
p = pv.Plotter()
lt = pv.LookupTable(values = pyvista_colors, value_range=[2,12])
lt.below_range_color = pv.Color('grey', opacity=0.5)
lt.above_range_color = pv.Color('grey', opacity=0.5)
#label = pv.Label("Cross section", position = [450, 300, 12])
rect = pv.Rectangle(points = [[0,max(ycorn)/2,0],[max(xcorn),max(ycorn)/2,0],[max(xcorn),max(ycorn)/2,max(zcorn)]])
p.add_mesh(horizontal_slice, scalars="facies", show_edges=False, cmap=my_cmap, clim=[2,12], show_scalar_bar=False, opacity=0.95)
p.add_mesh(vertical_slice, scalars="facies", show_edges=False, cmap=my_cmap, clim=[2,12], show_scalar_bar=False, opacity=0.95)
#p.add_actor(label)
p.add_mesh(mesh.tube(radius=0.3), color="crimson", opacity=0.6)
p.set_scale(1, 1, 30)
p.show()