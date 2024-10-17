import os
import numpy as np
import flopy
import sys

facies = np.load("facies_v7.npy")
# adding the dictionary of the hydraulic conductivity and reactivity:
unique_facies_indexes = np.unique(facies)
# TODO: Analyse this again
k_value_pair = { #hydraulic conductivity [m/s]
        2: 1e-5,
        4: 6e-6,
        6: 2e-6,
        7: 1e-6,
        8: 1e-7,
        9: 1e-7,
        10: 1e-7,
        11: 1.5e-5,
        12: 3e-4,
        21: 1e-8,
        31: 1e-8,
    }
# TODO: fix porosity with the reference from Cora's paper
porosity_pairs = { #no unit
    2: 0.75,
    4: 0.75,
    6: 0.75,
    7: 0.75,
    8: 0.8,
    9: 0.8,
    10: 0.20,
    11: 0.7,
    12: 0.25,
    21: 1.0,
    31: 1.0,
}
phi = np.array(list(porosity_pairs.values()))
TOC = np.array([2.5,5,8,18,24,34,1,2,1e-4, 0, 0])*1e-2 # %wt (proportion) total organic carbon
C_OM = 37.5*1e-2 #%wt (proportion) carbon content in organic matter
rho_om = 1350 #kg/m3 density of organic matter
rho_ms = 2710 #kg/m3 density of calcite
# /rho_s calculated based on expression from Ruehlmann et al. 2006
rho_s = (TOC/C_OM/rho_om + (1-TOC/C_OM)/rho_ms)**-1 #kg/m3
print(rho_s)
M_C = 12.01*1e-3 #kg/mol
C_content = TOC*rho_s*(1-phi)/M_C*(1/phi) #mol/m3
C_content = C_content*1e-3 #mol/L
print(C_content)
reactivity_pairs = dict(zip(unique_facies_indexes, C_content)) #mol/m3

vectorized_k = np.vectorize(k_value_pair.get)
vectorized_r = np.vectorize(reactivity_pairs.get)
vectorized_p = np.vectorize(porosity_pairs.get)

if __name__ == "__main__":
    sys.path.append(os.path.join("..","particle_track"))
    from particle_track import cumulative_cuda
    # loading datasets from the previous models:
    centroids = np.load("centroid_v7.npy")

    # Creating flopy models:
    k_array = vectorized_k(facies) #array of hydraulic conductivity
    r_array = vectorized_r(facies) #array of rel. reactivity
    p_array = vectorized_p(facies) #array of porosity

    # Loading the model
    ws = "ammer_v7"
    name = "ammer_V07"
    sim = flopy.mf6.MFSimulation.load(sim_name="mfsim", sim_ws=ws)
    gwf = sim.get_model(name)
    ## Reading output files
    head = gwf.output.head()
    head_array = head.get_data()
    grid = gwf.modelgrid
    # Calculating travel times and reactivity:
    centers = grid.xyzcellcenters
    X = centers[0]
    Y = centers[1]
    Z = centers[2]
    # Same size arrays:
    X = np.broadcast_to(X, Z.shape)
    Y = np.broadcast_to(Y, Z.shape)
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
    layers = np.ravel(layers)
    rows = np.ravel(rows)
    cols = np.ravel(cols)
    # Flattening arrays
    X = np.ravel(X)
    Y = np.ravel(Y)
    Z = np.ravel(Z)
    idomain = np.where(facies == 21, 0, np.where(facies == 31, 0, 1))
    X = X[idomain.ravel() == 1]
    Y = Y[idomain.ravel() == 1]
    Z = Z[idomain.ravel() == 1]
    layers = layers[idomain.ravel() == 1]
    rows = rows[idomain.ravel() == 1]
    cols = cols[idomain.ravel() == 1]
    # Particle centers:
    particle_centers = np.column_stack((X, Y, Z, layers, rows, cols))
    ct = cumulative_cuda(
        gwfmodel=gwf,
        model_directory=ws,
        particles_starting_location=particle_centers,
        porosity=p_array,
        reactivity=r_array,
    )
    print("Cumulative Reactivity: DONE")
    traveltimes = ct[:, 0]
    tt_array = np.zeros((nlay, nrow, ncol))
    tt_array[idomain == 1] = traveltimes
    cum_react = ct[:, 1]
    max_cum = np.max(cum_react)
    cm_array = np.zeros((nlay, nrow, ncol))
    cm_array[idomain == 1] = cum_react

    np.save(os.path.join(ws,"cum_react.npy"),cm_array)
    np.save(os.path.join(ws,"traveltimes.npy"),tt_array)
    np.save(os.path.join(ws,"facies.npy"),facies)
    np.save(os.path.join(ws,"f_.npy"),r_array)
