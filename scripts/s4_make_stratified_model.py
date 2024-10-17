import numpy as np
import flopy
import os
import sys
from s3_particle_track_TOC_model import vectorized_k, vectorized_r, vectorized_p
sys.path.append(os.path.join("..","particle_track"))
from particle_track import cumulative_cuda

if __name__ == "__main__":
    ws = "ammer_cake_v7"
    name = "ammer_cake"
    sim = flopy.mf6.MFSimulation(
        sim_name=name,
        exe_name="mf6",
        version="mf6",
        sim_ws=ws,
    )
    # Nam file
    model_nam_file = "{}.nam".format(name)
    # Groundwater flow object:
    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname=name,
        model_nam_file=model_nam_file,
        save_flows=True,
    )
    orig_facies = np.load("facies_v7.npy")
    unique_facies, counts = np.unique(orig_facies, return_counts=True)
    # Proportions converted to model layers:
    proportions = counts / np.sum(counts)
    # reorder index to match stratigraphic sequence:
    order = np.array([31, 9, 8, 7,6,4,2,12,11,10,21])
    index = [np.where(unique_facies == i)[0][0] for i in order]
    corrs_facies = unique_facies[index]
    assert np.all(corrs_facies == order)
    proportions = proportions[index]
    contacts = 9*np.cumsum(proportions)
    top_botm = np.concatenate([np.array([0]),contacts])

    # Grid properties:
    Lx = 900  # problem lenght [m]
    Ly = 600  # problem width [m]
    H = 9  # aquifer height [m]
    delx = 1.5  # block size x direction
    dely = 20  # block size y direction
    #delz = 0.07  # block size z direction
    nlay = top_botm.shape[0] - 1  # number of layers
    ncol = int(Lx / delx)  # number of columns
    nrow = int(Ly / dely)  # number of layers

    facies_cake = np.flip(corrs_facies)[:, np.newaxis, np.newaxis]*np.ones((nlay, nrow, ncol))
    idomain = np.where(facies_cake == 21, 0, np.where(facies_cake == 31, 0, 1))
    # Flopy Discretizetion Objects (DIS)
    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        xorigin=0.0,
        yorigin=0.0,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delx,
        delc=dely,
        top=H,
        botm=np.flip(top_botm)[1:],
        idomain=idomain,
    )

    # Calculate the proportion of each facies considering the layers have different thicknesses
    layer_thicknesses = np.diff(top_botm)
    total_thickness = np.sum(layer_thicknesses)
    facies_proportions = layer_thicknesses / total_thickness
    act_facies = np.unique(facies_cake)
    # make a layered model where the layers represent the volumetric ration of the V6 model

    # Checking if our assignment meets the measured proportions:
    assert np.all(act_facies == np.sort(corrs_facies))
    assert np.allclose(facies_proportions, proportions)
    k_array = vectorized_k(facies_cake)
    r_array = vectorized_r(facies_cake)
    p_array = vectorized_p(facies_cake)
 

    # Node property flow
    npf = flopy.mf6.ModflowGwfnpf(
        gwf,
        icelltype=0,  # This we define the model as confined
        k=k_array,
        save_specific_discharge=True,
    )


    # Acessing the grid
    grid = gwf.modelgrid
    tdis = flopy.mf6.ModflowTdis(
        sim, pname="tdis", time_units="SECONDS", nper=1, perioddata=[(1.0, 1, 1.0)]
    )
    ## Constant head
    chd_rec = []
    h = 12
    for lay in range(nlay):
        for row in range(nrow):
            # ((layer,row,col),head,iface)
            if idomain[lay, row, 0] == 1:
                chd_rec.append(((lay, row, 0), h, 1))
    h2 = h - ncol * delx * (2 / 750) #natural hydraulic gradient
    for lay in range(nlay):
        for row in range(nrow):
            # ((layer,row,col),head,iface)
            if idomain[lay, row, ncol-1] == 1:
                chd_rec.append(((lay, row, ncol - 1), h2, 2))
    chd = flopy.mf6.ModflowGwfchd(
        gwf,
        auxiliary=[("iface",)],
        stress_period_data=chd_rec,
        print_input=True,
        print_flows=True,
        save_flows=True,
    )
    # Flopy initial Conditions
    start = h * np.ones((nlay, nrow, ncol))
    ic = flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=start)
    # Output control and Solver
    headfile = "{}.hds".format(name)
    head_filerecord = [headfile]
    budgetfile = "{}.cbb".format(name)
    budget_filerecord = [budgetfile]
    saverecord = [("HEAD", "ALL"), ("BUDGET", "ALL")]
    printrecord = [("HEAD", "LAST")]
    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        saverecord=saverecord,
        head_filerecord=head_filerecord,
        budget_filerecord=budget_filerecord,
        printrecord=printrecord,
    )
    ims = flopy.mf6.ModflowIms(
        sim,
        pname="ims",
        complexity="SIMPLE",
        # linear_acceleration="BICGSTAB",
        outer_maximum=10,
        inner_maximum=20000,
        outer_dvclose=1e-5,
        inner_dvclose=1e-6,
        rcloserecord=[1e-6, "STRICT"],
    )
    # Solving
    sim.write_simulation()
    sim.check()
    success, buff = sim.run_simulation()
    if not success:
        raise Exception("MODFLOW 6 did not terminate normally.")
    print(buff)
    ## Reading output files

    head = gwf.output.head()
    head_array = head.get_data()
    grid = gwf.modelgrid

    # Calculating travel times and reactivity:
    centers = grid.xyzcellcenters
    X = centers[0]
    Y = centers[1]
    Z = centers[2]
    X = np.broadcast_to(X, Z.shape)
    Y = np.broadcast_to(Y, Z.shape)
    # Extracting indexes:
    layers = np.arange(0, nlay)
    layers = layers[:, np.newaxis, np.newaxis]
    layers = layers * np.ones(Z.shape)
    rows = np.arange(0, nrow)
    rows = rows[np.newaxis, :, np.newaxis]
    rows = rows * np.ones(Z.shape)
    cols = np.arange(0, ncol)
    cols = cols[np.newaxis, np.newaxis, :]
    cols = cols * np.ones(Z.shape)
    ex_layers = np.ravel(layers[:,:,-1])
    ex_rows = np.ravel(rows[:,:,-1])
    ex_cols = np.ravel(cols[:,:,-1])
    ex_x = np.ravel(X[:,:,-1])
    ex_y = np.ravel(Y[:,:,-1])
    ex_z = np.ravel(Z[:,:,-1])
    ex_particle_centers = np.column_stack((ex_x, ex_y, ex_z, ex_layers, ex_rows, ex_cols))
    layers = np.ravel(layers)
    rows = np.ravel(rows)
    cols = np.ravel(cols)
    # Flattening arrays
    X = np.ravel(X)
    Y = np.ravel(Y)
    Z = np.ravel(Z)
    # Particle centers:
    particle_centers = np.column_stack((X, Y, Z, layers, rows, cols))
    ct = cumulative_cuda(
        gwfmodel=gwf,
        model_directory=ws,
        particles_starting_location=particle_centers,
        porosity=p_array,
        reactivity=r_array,
        debug=False,
    )
    print("Cumulative Reactivity: DONE")
    traveltimes = ct[:, 0]
    traveltimes = np.reshape(traveltimes, k_array.shape)
    cum_react = ct[:, 1]
    
    max_cum = np.max(cum_react)

    # reshaping:
    cum_react = np.reshape(cum_react, k_array.shape)
    traveltimes = np.reshape(traveltimes, k_array.shape)
    np.save(os.path.join(ws,"cum_react.npy"),cum_react)
    np.save(os.path.join(ws,"traveltimes.npy"),traveltimes)
    np.save(os.path.join(ws,"facies.npy"),facies_cake)
    np.save(os.path.join(ws,"f_.npy"),r_array)