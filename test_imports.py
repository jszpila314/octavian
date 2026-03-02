from mpi4py import MPI
import octavian
import os
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nsplit = comm.Get_size()

path_to_snapshot = "/disk04/rad/sim/m25n512/s50/snap_m25n512_062.hdf5"
path_to_filtered_output = "/home/jpduminy/octavian-analysis/split-snapshot"
path_to_filtered_analysis = "/home/jpduminy/octavian-analysis/finished-snapshot"
path_to_output = "/home/jpduminy/octavian-analysis"
path_to_config = "/home/jpduminy/octavian_scripts/octavian-1/config.yaml"

if rank == 0:
    import caesar
    print(f"caesar imported successfully.")
    import pygadgetreader
    print(f"pygadgetreader imported successfully.")
    import fsps
    print(f"fsps imported successfully.")
    import octavian
    print(f"octavian imported successfully.")
    print(f"Packages imported successfully!")

    # remove files from previous runs
    for i in range(nsplit):
        old_file = f"{path_to_filtered_output}_{i}.hdf5"
        if os.path.exists(old_file):
            os.remove(old_file)
    
    print(f"Attempting to filter snapshot...")
    octavian.filter_snapshot(path_to_snapshot, path_to_filtered_output, nsplit=nsplit)
    print("Filtering complete.")

comm.Barrier()
                        
octavian.mpirun(path_to_filtered_output, path_to_filtered_analysis, path_to_config)

if rank == 0:
    print(f"Success.")
