
import flopy
import numpy as np

if __name__ == "__main__":

    # loading datasets from the previous models:
    centroids = np.load("centroid_v7.npy")
    facies = np.load("facies_v7.npy")

    # I want to run MODFLOW on the HyVR model and take slices in the end to calculate the cumulative OC residence and flow rate:
    # index of the slices:
    slices_ind = np.linspace(10,facies.shape[2], 5).astype(np.int32)
    # column indexes: lower range because the model is too long:

    unique_facies_indexes = np.unique(facies)

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

    vectorized_k = np.vectorize(k_value_pair.get)

    # Creating flopy models:
    k_array = vectorized_k(facies) #array of hydraulic conductivity
    # Model creation:
    name = "ammer_V07"
    ws = "ammer_v7"
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
    # Grid properties:
    Lx = 900  # problem lenght [m]
    Ly = 600  # problem width [m]
    H = 9  # aquifer height [m]
    delx = 1.5  # block size x direction
    dely = 1.5  # block size y direction
    delz = 0.2  # block size z direction
    nlay = int(H / delz)
    ncol = int(Lx / delx)  # number of columns
    nrow = int(Ly / dely)  # number of layers

    idomain = np.where(facies == 21, 0, np.where(facies == 31, 0, 1))
    # Flopy Discretizetion Objects (DIS)
    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        xorigin=0.0,
        yorigin=0.0,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=dely,
        delc=delx,
        top=H,
        botm=np.arange(H - delz, 0 - delz, -delz),
        idomain=idomain,
    )

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
    #Q = 6.7e-5*(2/750)
    #flow_per_cell_in = Q/np.sum(idomain[:,:,0])
    #flow_per_cell_out = (-1)*Q/np.sum(idomain[:,:,ncol-1])
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
        print_option="ALL",
        pname="ims",
        complexity="SIMPLE",
        linear_acceleration="CG",
        outer_maximum=10,
        inner_maximum=2000,
        outer_dvclose=1e-6,
        inner_dvclose=1e-8,
        rcloserecord=[1e-7, "STRICT"],
    )
    # Solving
    sim.write_simulation()
    sim.check()
    success, buff = sim.run_simulation()
    if not success:
        raise Exception("MODFLOW 6 did not terminate normally.")
    print(buff)
