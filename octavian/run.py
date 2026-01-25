from time import perf_counter

from octavian.data_manager import DataManager, save_group_properties
from octavian.utils import wrap_positions
from octavian.halo_finder import run_fof6d
from octavian.group_properties_calc import calculate_group_properties

from yaml import safe_load


def run(snapshot: str, outfile: str, configfile: str):

  # mpi functionality; check whether it is installed and if not use serial case
  try: 
    from mpi4py import MPI
    comm = MPI_COMM_WORLD()
    rank = comm.Get_rank()
  except ImportError:
    comm = None
    rank = 0

  if rank == 0:
    print(f"Initialising Data Manager...")

  with open(configfile, 'r') as f:
    config = safe_load(f)

  config['Tlim'] = float(config['Tlim'])

  t1 = perf_counter()
  data_manager = DataManager(snapshot, config)
  t2 = perf_counter()
  print(f'Done in {t2-t1:.2f} seconds.')

  data_manager.load_halo_ids()
  data_manager.add_ptype_columns()

  print('Wrapping positions...')
  t1 = perf_counter()
  wrap_positions(data_manager)
  t2 = perf_counter()
  print(f'Done in {t2-t1:.2f} seconds.')

  print('Running FOF6D...')
  t1 = perf_counter()
  run_fof6d(data_manager)
  t2 = perf_counter()
  print(f'Done in {t2-t1:.2f} seconds.')

  print('Calculating group properties...')
  t1 = perf_counter()
  data_manager.initialise_group_data()
  calculate_group_properties(data_manager)
  t2 = perf_counter()
  print(f'Done in {t2-t1:.2f} seconds.')

  print('Saving datasets...')
  t1 = perf_counter()
  save_group_properties(data_manager, outfile)
  t2 = perf_counter()
  print(f'Done in {t2-t1:.2f} seconds.')