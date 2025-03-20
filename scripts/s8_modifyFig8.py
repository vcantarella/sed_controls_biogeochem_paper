import flopy
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmcrameri as cm
import matplotlib.ticker as mticker
from s3_particle_track_TOC_model import vectorized_p
from PIL import Image


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
cr = np.load(os.path.join(ws, "cum_react.npy"))
#cr[cr == np.inf] = np.nan
#Advective breakthrough time:
c_in = 80*1e-3/(14+3*16) # mg/L
gamma_n = 4/5
factor = 1/(gamma_n*c_in)
cr = cr*factor
cr = cr/(3600*24*365) #years
# instantaneous reactable concentration of nitrate
colors_full = {2:"#cec47f",4:"#c9c186",6:"#b0b468",7:"#c8b23d",8:"#323021", 9:"#5a1919", 10:"#3c5579", 11:"#FCF2A8", 12:"#b7b7b7"}
keys = (np.array(list(colors_full.keys()))-2)/10
cmap = mpl.colors.ListedColormap(list(colors_full.values()))
bounds = [2,4,6,7,8,9,10,11,12,13]
# Create a normalization object
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

hatches = {11:"o",2:".",4:"",6:"..",7:"..",8:"",9:"",12:"oo", 10:""}
labels = {11:"T1",2:"T2",4:"T4",6:"T6",7:"T7",8:"T8",9:"T9",12:"G1",10:"C1"}
f_ = np.load(os.path.join(ws, "f_.npy"))
k = gwf.npf.k.array
tts = np.load(os.path.join(ws, "traveltimes.npy"))
tts = tts/(3600*24*365) #years
tts[tts == np.inf] = np.nan
budget = gwf.output.budget()
head = gwf.output.head().get_data()
flow_ja_face = budget.get_data(text="FLOW-JA-FACE")[0]
structured_face_flows = flopy.mf6.utils.postprocessing.get_structured_faceflows(
    flow_ja_face,
    grb_file=os.path.join(ws, gwf.name + ".dis.grb"),
    verbose=True,
)
frf, fff, flf = structured_face_flows

qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(
    (frf, fff, flf), gwf, head=head)
porosity = vectorized_p(facies)
assert np.unique(porosity).size > 2
vx = qx / porosity
vy = qy / porosity
vz = qz / porosity
v = np.sqrt(vx**2 + vy**2 + vz**2)
# make a slice mask array:
mask = np.zeros_like(facies)
# sample 100 points systematically in the row and vertical direction
nlay, nrow, ncol = facies.shape
step_lay = int(nlay / 8)
step_col = int(ncol / 30)
col_indices = np.arange(0, ncol, step_col)
lay_indices = np.arange(0, nlay, step_lay)
for lay in lay_indices:
    mask[lay, :, col_indices] = np.where(v[lay, :, col_indices] > 0, 1, 0)
vx[mask == 0] = np.nan
vy[mask == 0] = np.nan
vz[mask == 0] = np.nan


fig1, axs_d = plt.subplot_mosaic([["a. 3D view", "a. 3D view"],
                                  ["b. facies stratified", "c. facies 3-D facies-based"],
                                  ["d. $K$ stratified", "e. $K$ 3-D facies-based"],
                                  ["f. TOC stratified", "g. TOC 3-D facies-based"]],
                                   constrained_layout=True,figsize=(7.2,6.4))
fig2, axs_d2 = plt.subplot_mosaic([["a. velocities stratified", "b. velocities 3-D facies-based"],
                                   ["c. travel-times stratified", "d. travel-times 3-D facies-based"],
                                   ["e. $\\tau_{inst}$ stratified", "f. $\\tau_{inst}$ 3-D facies-based"]],
                                   constrained_layout=True,figsize=(7.2,4.8))


#facies
aspect = 30
ax1 = axs_d["c. facies 3-D facies-based"]
ax1.set_aspect(aspect=aspect, share = True)
xsect1 = flopy.plot.PlotCrossSection(
    ax = ax1,
    model=gwf, line={"Row": int(nrow/2)}, geographic_coords=True
)
csa1 = xsect1.plot_array(facies, alpha=0.95, cmap=cmap, norm =norm, masked_values=[21,31])
#cbar = fig1.colorbar(csa1, ticks=list(colors_full.keys()), label="Facies", shrink=0.7)
#cbar.ax.set_yticklabels(list(colors_full.keys()))  # set the colorbar ticks to be the keys of the dictionary

# fig1.colorbar(csa1, label="Facies", shrink=0.6)
ax2 = axs_d["e. $K$ 3-D facies-based"]
ax2.set_aspect(aspect=aspect, share = True)
xsect2 = flopy.plot.PlotCrossSection(
    ax=ax2,
    model=gwf, line={"Row": int(nrow/2)}, geographic_coords=True
)
csa2 = xsect2.plot_array(k, alpha=0.95, cmap=cm.cm.nuuk, norm=mpl.colors.LogNorm())
fig1.colorbar(csa2, label="$k$ [m/s]", shrink=0.6)
ax3 = axs_d["g. TOC 3-D facies-based"]
ax3.set_aspect(aspect=aspect)
xsect3 = flopy.plot.PlotCrossSection(
    ax=ax3,
    model=gwf, line={"Row": int(nrow/2)}, geographic_coords=True
)
csa3 = xsect3.plot_array(f_, alpha=0.95, cmap=cm.cm.oslo_r,)# norm=mpl.colors.LogNorm())
ax3.set_xlabel("Distance [m]")
# def formatter_TOC(x, pos):
#     return f"{x:.1e}"
fig1.colorbar(csa3, label="TOC [molC/L]", shrink=0.6)#, format=mticker.FuncFormatter(formatter_TOC))

axx1 = axs_d2["b. velocities 3-D facies-based"]
axx1.set_aspect(aspect=aspect)
xsectx1 = flopy.plot.PlotCrossSection(
    ax=axx1,
    model=gwf, line={"Row": int(nrow/2)}, geographic_coords=True
)
quiver = xsectx1.plot_vector(
    vx,
    vy,
    vz,
    hstep=1,
    normalize=True,
    masked_values=[np.nan,-np.inf,np.inf],
    color="darkred",
    scale=60,
    headwidth=5,
    headlength=3,
    headaxislength=3,
    zorder=10,
)

csax1 = xsectx1.plot_array(v, alpha=0.55, cmap=cm.cm.davos_r, norm=mpl.colors.LogNorm())
fig2.colorbar(csax1, label="$|v|$ [m/s]", shrink=0.6)
axx2 = axs_d2["d. travel-times 3-D facies-based"]
axx2.set_aspect(aspect=aspect)
xsectx2 = flopy.plot.PlotCrossSection(
    ax=axx2,
    model=gwf, line={"Row": int(nrow/2)}, geographic_coords=True
)

ttmin = 20
ttmax = np.nanmax(tts)
def formatter_t(x, pos):
    if x <= ttmin:
        return f"<{x:.0e}"
    else:
        return f"{x:.0e}"
csax2 = xsectx2.plot_array(tts, alpha=0.95, cmap=cm.cm.roma_r, vmin=ttmin, vmax=ttmax, norm=mpl.colors.LogNorm())
fig2.colorbar(csax2, label="Travel-time [years]", shrink=0.6, format=mticker.FuncFormatter(formatter_t))
axx3 = axs_d2["f. $\\tau_{inst}$ 3-D facies-based"]
axx3.set_aspect(aspect=aspect)
xsectx3 = flopy.plot.PlotCrossSection(
    ax=axx3,
    model=gwf, line={"Row": int(nrow/2)}, geographic_coords=True
)
vmax = cr.max()
vmin = 1e4
csax3 = xsectx3.plot_array(cr, alpha=0.95, cmap=cm.cm.vik, vmin=vmin, vmax=vmax, norm=mpl.colors.LogNorm())
axx3.set_xlabel("Distance [m]")
def formatter(x, pos):
    if x <= vmin:
        return f"<{x:.0e}"
    else:
        return f"{x:.0e}"

fig2.colorbar(csax3, label="$\\tau_{inst}$ [years]", shrink=0.6, format=mticker.FuncFormatter(formatter))


## loading the average thickness model:
ws = "ammer_cake_v7"
ammer_model = flopy.mf6.MFSimulation.load(sim_ws=ws)
gwf = ammer_model.get_model("ammer_cake")
nrow = gwf.modelgrid.nrow
cr = np.load(os.path.join(ws, "cum_react.npy"))
cr = cr*factor
cr = cr/(3600*24*365) #years
facies = np.load(os.path.join(ws, "facies.npy"))
facies = facies.astype(np.int16)
tts = np.load(os.path.join(ws, "traveltimes.npy"))
f_ = np.load(os.path.join(ws, "f_.npy"))
k = gwf.npf.k.array
f_ = np.load(os.path.join(ws, "f_.npy"))
k = gwf.npf.k.array
tts = np.load(os.path.join(ws, "traveltimes.npy"))
tts = tts/(3600*24*365) #years
budget = gwf.output.budget()
head = gwf.output.head().get_data()
flow_ja_face = budget.get_data(text="FLOW-JA-FACE")[0]
structured_face_flows = flopy.mf6.utils.postprocessing.get_structured_faceflows(
    flow_ja_face,
    grb_file=os.path.join(ws, gwf.name + ".dis.grb"),
    verbose=True,
)
frf, fff, flf = structured_face_flows

qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(
    (frf, fff, flf), gwf, head=head)
porosity = vectorized_p(facies)
vx = qx / porosity
vy = qy / porosity
vz = qz / porosity
v = np.sqrt(vx**2 + vy**2 + vz**2)
# make a slice mask array:
mask = np.zeros_like(facies)
# sample 100 points systematically in the row and vertical direction
nlay, nrow, ncol = facies.shape
step_lay = int(nlay / 8)
step_col = int(ncol / 30)
col_indices = np.arange(0, ncol, step_col)
lay_indices = np.arange(0, nlay, step_lay)
for lay in lay_indices:
    mask[lay, :, col_indices] = 1
vx[mask == 0] = np.nan
vy[mask == 0] = np.nan
vz[mask == 0] = np.nan
#fig = plt.figure(figsize=(10,20))
#facies
ax4 = axs_d["b. facies stratified"]
ax4.set_aspect(aspect=aspect)
xsect4 = flopy.plot.PlotCrossSection(
    ax=ax4, model=gwf, line={"Row": int(nrow/2)}, geographic_coords=True
)
csa4 = xsect4.plot_array(facies, alpha=0.95, cmap=cmap, norm=norm)
ax4.set_ylabel("Depth [m]")
ax5 = axs_d["d. $K$ stratified"]
ax5.set_aspect(aspect=aspect)
xsect5 = flopy.plot.PlotCrossSection(
    ax=ax5,model=gwf, line={"Row": int(nrow/2)}, geographic_coords=True
)
csa5 = xsect5.plot_array(k, alpha=0.95, cmap=cm.cm.nuuk, norm=mpl.colors.LogNorm())
ax5.set_ylabel("Depth [m]")
ax6 = axs_d["f. TOC stratified"]
ax6.set_aspect(aspect=aspect)
xsect6 = flopy.plot.PlotCrossSection(
    ax=ax6, model=gwf, line={"Row": int(nrow/2)}, geographic_coords=True
)
csa6 = xsect6.plot_array(f_, alpha=0.95, cmap=cm.cm.oslo_r,)# norm=mpl.colors.LogNorm())
ax6.set_xlabel("Distance [m]")
ax6.set_ylabel("Depth [m]")
axx4 = axs_d2["a. velocities stratified"]
axx4.set_aspect(aspect=aspect)
xsectx4 = flopy.plot.PlotCrossSection(
    ax=axx4,model=gwf, line={"Row": int(nrow/2)}, geographic_coords=True
)
quiverx4 = xsectx4.plot_vector(
    vx,
    vy,
    vz,
    hstep=1,
    normalize=True,
    color="darkred",
    scale=60,
    headwidth=5,
    headlength=3,
    headaxislength=3,
    zorder=10,
)
csax4 = xsectx4.plot_array(v, alpha=0.55, cmap=cm.cm.davos_r, norm=mpl.colors.LogNorm())
#fig2.colorbar(csa, label="Velocity magnitude [m/s]", shrink=0.8)
axx4.set_ylabel("Depth [m]")
axx5 = axs_d2["c. travel-times stratified"]
axx5.set_aspect(aspect=aspect)
xsectx5 = flopy.plot.PlotCrossSection(
    ax=axx5, model=gwf, line={"Row": int(nrow/2)}, geographic_coords=True
)
csax5 = xsectx5.plot_array(tts, alpha=0.95, cmap=cm.cm.roma_r, vmin=ttmin, vmax=ttmax, norm=mpl.colors.LogNorm())
axx5.set_ylabel("Depth [m]")
axx6 = axs_d2["e. $\\tau_{inst}$ stratified"]
axx6.set_aspect(aspect=aspect)
xsectx6 = flopy.plot.PlotCrossSection(
    ax=axx6, model=gwf, line={"Row": int(nrow/2)}, geographic_coords=True
)
csax6 = xsectx6.plot_array(cr, alpha=0.95, cmap=cm.cm.vik, vmin=vmin, vmax=vmax, norm=mpl.colors.LogNorm())
axx6.set_xlabel("Distance [m]")
axx6.set_ylabel("Depth [m]")
# fig2.colorbar(csa, label="Cumulative Reactivity [molC s L^-1]", shrink=0.8)



for label, ax in axs_d.items():

    ax.set_ylim(0, 9)
    ax.set_yticks(np.arange(0, 9, 3))
    h_ticks = ax.get_yticks()
    max_height = 9  # This assumes the maximum height is the largest tick
    depth_ticks = 9 - h_ticks
    depth_tick_labels = [f"{depth:.1f}" for depth in depth_ticks]
    ax.set_yticklabels(depth_tick_labels)
    ax.set_title(label, loc='left', fontsize='medium')

for label, ax in axs_d2.items():
    ax.set_ylim(0, 9)  
    ax.set_yticks(np.arange(0, 9, 3))
    h_ticks = ax.get_yticks()
    max_height = 9  # This assumes the maximum height is the largest tick
    depth_ticks = 9 - h_ticks
    depth_tick_labels = [f"{depth:.1f}" for depth in depth_ticks]
    ax.set_yticklabels(depth_tick_labels)
    ax.set_title(label, loc='left', fontsize='medium')


ax3d = axs_d["a. 3D view"]
img = Image.open("3D_plot_click_edit.png")
# img = np.asarray(Image.open("3D_plot_click.png"))
ax3d.imshow(img, aspect='auto')
ax3d.axis("off")
# Force update before other plots are added
plt.draw()

fig1.show()
fig1.savefig("model_slice_3D.png", dpi=1000, bbox_inches='tight')