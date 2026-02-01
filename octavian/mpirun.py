from mpi4py import MPI
from octavian.run import run

def mpirun(base_snapshot: str, base_outfile: str, configfile: str):
    """
    Runs Octavian in MPI configuration.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # each rank accesses its own part of the snapshot
    snapshot = f"{base_snapshot}_{rank}.hdf5"
    output = f"{base_outfile}_{rank}.hdf5"

    if rank == 0:
        print(f"Running Octavian with {size} nodes.")

    run(snapshot, output, configfile)

    if rank == 0:
        print(f"All ranks complete.")