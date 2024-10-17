import numpy as np
import flopy
import os
import sys
import matplotlib.pyplot as plt
from s3_particle_track_TOC_model import vectorized_k, vectorized_r, vectorized_p
from flopy.mf6.utils.postprocessing import get_structured_faceflows
sys.path.append(os.path.join("..","particle_track"))
from particle_track import cumulative_cuda

#Advective breakthrough time:
c_in = 80*1e-3/(14+3*16) # mg/L to mol/L
gamma_n = 4/5
factor = 1/(gamma_n*c_in)


# loading datasets from the previous models:
facies = np.load("facies_v7.npy")
k_array = vectorized_k(facies)
r_array = vectorized_r(facies)
p_array = vectorized_p(facies)
# loading the model:
unique = np.unique(facies, return_counts=True)
unique[1]/np.sum(unique[1])
ws = "ammer_v7"
## loading the model:
ammer_model = flopy.mf6.MFSimulation.load(sim_ws=ws)
gwf = ammer_model.get_model("ammer_V07")
budget = gwf.output.budget()
flow_ja_face = budget.get_data(text="FLOW-JA-FACE")[0]
structured_face_flows = get_structured_faceflows(
    flow_ja_face,
    grb_file=os.path.join(ws, gwf.name + ".dis.grb"),
    verbose=True,
)
frf = structured_face_flows[0]
# Calculating travel times and reactivity:
grid = gwf.modelgrid
centers = grid.xyzcellcenters
X = centers[0]
Y = centers[1]
Z = centers[2]
# Same size arrays:
X = np.broadcast_to(X, Z.shape)
Y = np.broadcast_to(Y, Z.shape)
# Selecting last rows:
X = X[:,:,-1]
Y = Y[:,:,-1]
Z = Z[:,:,-1]
nlay = gwf.modelgrid.nlay
nrow = gwf.modelgrid.nrow
ncol = gwf.modelgrid.ncol
# Extracting indexes:
layers = np.arange(0, nlay)
layers = layers[:, np.newaxis, np.newaxis]
layers = layers * np.ones(centers[2].shape)
rows = np.arange(0, nrow)
rows = rows[np.newaxis, :, np.newaxis]
rows = rows * np.ones(centers[2].shape)
cols = np.arange(0, ncol)
cols = cols[np.newaxis, np.newaxis, :]
cols = cols * np.ones(centers[2].shape)
idomain = np.where(facies == 21, 0, np.where(facies == 31, 0, 1))
idomain = idomain[:,:,-1].ravel()
# Selecting the last rows:
layers = layers[:,:,-1]
rows = rows[:,:,-1]
cols = cols[:,:,-1]
# Flattening arrays
X = np.ravel(X)
Y = np.ravel(Y)
Z = np.ravel(Z)
layers = np.ravel(layers)
rows = np.ravel(rows)
cols = np.ravel(cols)
# adding the half cell size to the x value, so the point is at the exit of the cell
delr = gwf.dis.delr.array
X = X + delr[-1]/2
# Particle centers:
particle_centers = np.column_stack((X, Y, Z, layers, rows, cols))
particle_centers = particle_centers[idomain == 1,:]
ct = cumulative_cuda(
    gwfmodel=gwf,
    model_directory=ws,
    particles_starting_location=particle_centers,
    porosity=p_array,
    reactivity=r_array,
)
ct = ct/(3600*24*365)
print("Cumulative Reactivity: DONE")
traveltimes = ct[:, 0]
cum_react = ct[:, 1]
adv_breaktime_v6 = cum_react*factor
max_cum = np.max(cum_react)
bound_flow = budget.get_data(text="CHD")[0]
for i in range(bound_flow.shape[0]):
    iface = bound_flow[i]["IFACE"]
    q = bound_flow[i]["q"]
    node = bound_flow[i]["node"]-1
    lrc = grid.get_lrc([node])[0]
    if iface == 2:
        frf[lrc[0],lrc[1],lrc[2]] = -q
# Area of the grid cells in the outflow direction:
delc = gwf.dis.delc.array
top_botm = grid.top_botm
delz = -np.diff(top_botm, axis=0)
area = delc[np.newaxis,:,np.newaxis]*delz
exiting_area = area[:,:,-1]
assert np.allclose(np.sum(exiting_area),9*600)
# Flows in the outflow direction
exiting_idomain = idomain
exiting_flows_v6 = frf[:,:,-1]
exiting_flows_v6 = exiting_flows_v6.ravel()[exiting_idomain == 1]
exiting_porosity = p_array[:,:,-1]
mean_toc_exposure_v6 = cum_react/traveltimes

# DOing the same for the layer cake model:
ws = "ammer_cake_v7"
## loading the model:
ammer_model = flopy.mf6.MFSimulation.load(sim_ws=ws)
gwf = ammer_model.get_model("ammer_cake")
budget = gwf.output.budget()
flow_ja_face = budget.get_data(text="FLOW-JA-FACE")[0]
structured_face_flows = get_structured_faceflows(
    flow_ja_face,
    grb_file=os.path.join(ws, gwf.name + ".dis.grb"),
    verbose=True,
)
frf = structured_face_flows[0]
# Calculating travel times and reactivity:
grid = gwf.modelgrid
centers = grid.xyzcellcenters
X = centers[0]
Y = centers[1]
Z = centers[2]
# loading the cake facies:
facies_cake = np.load(os.path.join(ws, "facies.npy"))
k_array = vectorized_k(facies_cake)
r_array = vectorized_r(facies_cake)
p_array = vectorized_p(facies_cake)

# Same size arrays:
X = np.broadcast_to(X, Z.shape)
Y = np.broadcast_to(Y, Z.shape)
# Selecting last rows:
X = X[:,:,-1]
Y = Y[:,:,-1]
Z = Z[:,:,-1]
nlay = gwf.modelgrid.nlay
nrow = gwf.modelgrid.nrow
ncol = gwf.modelgrid.ncol
# Extracting indexes:
layers = np.arange(0, nlay)
layers = layers[:, np.newaxis, np.newaxis]
layers = layers * np.ones(centers[2].shape)
rows = np.arange(0, nrow)
rows = rows[np.newaxis, :, np.newaxis]
rows = rows * np.ones(centers[2].shape)
cols = np.arange(0, ncol)
cols = cols[np.newaxis, np.newaxis, :]
cols = cols * np.ones(centers[2].shape)
idomain = np.where(facies_cake == 21, 0, np.where(facies_cake == 31, 0, 1))
idomain = idomain[:,:,-1].ravel()
# Selecting the last rows:
layers = layers[:,:,-1]
rows = rows[:,:,-1]
cols = cols[:,:,-1]
# Flattening arrays
X = np.ravel(X)
Y = np.ravel(Y)
Z = np.ravel(Z)
layers = np.ravel(layers)
rows = np.ravel(rows)
cols = np.ravel(cols)
# adding the half cell size to the x value, so the point is at the exit of the cell
delr = gwf.dis.delr.array
X = X + delr[-1]/2
# Particle centers:
particle_centers = np.column_stack((X, Y, Z, layers, rows, cols))
particle_centers = particle_centers[idomain == 1,:]
ct = cumulative_cuda(
    gwfmodel=gwf,
    model_directory=ws,
    particles_starting_location=particle_centers,
    porosity=p_array,
    reactivity=r_array,
)
ct = ct/(3600*24*365)
traveltimes = ct[:, 0]
# traveltimes = np.reshape(traveltimes, k_array.shape)
cum_react_cake = ct[:, 1]
adv_breaktime_cake = cum_react_cake*factor #in years
max_cum_cake = np.max(cum_react_cake)

bound_flow = budget.get_data(text="CHD")[0]
for i in range(bound_flow.shape[0]):
    iface = bound_flow[i]["IFACE"]
    q = bound_flow[i]["q"]
    node = bound_flow[i]["node"]-1
    lrc = grid.get_lrc([node])[0]
    if iface == 2:
        frf[lrc[0],lrc[1],lrc[2]] = -q
delc = gwf.dis.delc.array
top_botm = grid.top_botm
delz = -np.diff(top_botm, axis=0)
area = delc[np.newaxis,:,np.newaxis]*delz
exiting_area = area[:,:,-1]
np.sum(exiting_area)
exiting_idomain = idomain
exiting_flows_cake = frf[:,:,-1]
exiting_flows_cake = exiting_flows_cake.ravel()[exiting_idomain == 1]
exiting_porosity = p_array[:,:,-1]

mean_toc_exposure_cake = cum_react_cake/traveltimes

#Uniform case:

unique_facies, counts = np.unique(facies, return_counts=True)
# Proportions converted to model layers:
proportions = counts / np.sum(counts)
# reorder index to match stratigraphic sequence:
order = np.array([9, 8, 7,6,4,2,12,11,10])
index = [np.where(unique_facies == i)[0][0] for i in order]
corrs_facies = unique_facies[index]
assert np.all(corrs_facies == order)
proportions = proportions[index]
contacts = np.cumsum(proportions)*9
top_botm = np.concatenate([np.array([0]),contacts])
k_array = vectorized_k(corrs_facies)
r_array = vectorized_r(corrs_facies)
p_array = vectorized_p(corrs_facies)
mean_k = np.sum(k_array*proportions)/np.sum(proportions)
mean_r = np.sum(r_array*proportions)/np.sum(proportions)
mean_p = np.sum(p_array*proportions)/np.sum(proportions)
head = gwf.output.head().get_alldata()[0]
head[head == 1e30] = np.nan
delta_h = np.nanmax(head) - np.nanmin(head)
H = top_botm.max()
Lx = 900
Ly = 600
i = delta_h/(Lx-1.5)
A = Ly*H
Q = mean_k*i*A
q = mean_k*i
v = q/mean_p
exp_time = mean_r*(Lx)/v/(3600*24*365)
exp_time = exp_time*factor
adv_break_times_uniform = exp_time*v
exp_vel_uniform = exp_time*1e3*v


## Plotting advective breakthrough times scaled by flux:
sorted_adv_v6 = np.sort(adv_breaktime_v6)
argsort_v6 = np.argsort(adv_breaktime_v6)
sorted_flows_v6 = exiting_flows_v6.ravel()[argsort_v6]
P_v6 = np.cumsum(sorted_flows_v6)/np.sum(sorted_flows_v6)
q_s = sorted_adv_v6
p_s = P_v6

sorted_adv_cake = np.sort(adv_breaktime_cake)
argsort_cake = np.argsort(adv_breaktime_cake)
sorted_flows_cake = exiting_flows_cake.ravel()[argsort_cake]
P_cake = np.cumsum(sorted_flows_cake)/np.sum(sorted_flows_cake)
q_l = sorted_adv_cake
p_l = P_cake

mean_l = np.sum(adv_breaktime_cake*exiting_flows_cake.ravel())/np.sum(exiting_flows_cake.ravel())
mean_s = np.sum(adv_breaktime_v6*exiting_flows_v6.ravel())/np.sum(exiting_flows_v6)
p_mean_l = p_l[np.argmin(np.abs(q_l-mean_l))]
p_mean_s = p_s[np.argmin(np.abs(q_s-mean_s))]


fig, ax = plt.subplots(figsize=(7.9,3.9))
lay_color = "#89BA5E"
sed_color = "#5E89BA"
ax.plot([exp_time, exp_time], [0, 1], color="#BA5E89", label="uniform model", linestyle="-", lw=2.1)
ax.plot(q_l, p_l, label="stratified model", color = lay_color, lw=2.5)
ax.plot([mean_l, mean_l], [0, p_mean_l], color=lay_color,
 label="mean, stratified model", linestyle="--", lw = 2.4)
ax.plot(q_s, p_s, label="3-D facies based model", color = sed_color, lw=2.5)
ax.plot([mean_s, mean_s], [0, p_mean_s], color=sed_color,
 label="mean, 3-D facies based model", linestyle="--", lw = 2.4)
ax.set_xlim(1e0, 5e8)
ax.set_ylim(0, 1)
ax.set_xscale("log")
ax.set_xlabel("$\\tau_{inst} $ [years]")
ax.set_ylabel("flux-weighted CDF")
ax.legend(loc="upper left")
fig.show()
fig.savefig("adv_break_cdfv7_o2.png", dpi=1000, bbox_inches="tight")

