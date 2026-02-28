from time import perf_counter

from octavian.data_manager import DataManager, save_group_properties
from octavian.utils import wrap_positions, merge_catalogues
from octavian.halo_finder import run_fof6d
from octavian.group_properties_calc import calculate_group_properties, get_particle_lists

from yaml import safe_load

import memray
import os


def run(snapshot: str, outfile: str, configfile: str, comm=None):
  with open(configfile, 'r') as f:
    config = safe_load(f)

  config['Tlim'] = float(config['Tlim'])

  t1 = perf_counter()
  data_manager = DataManager(snapshot, config, comm=comm)
  t2 = perf_counter()
  print(f'Done in {t2-t1:.2f} seconds.')

  memray_file = f"memray_rank_{data_manager.rank}.bin"
  if os.path.exists(memray_file):
      os.remove(memray_file)

  with memray.Tracker(f"memray_rank_{data_manager.rank}.bin", native_traces=True):
    data_manager.load_halo_ids()
    data_manager.add_ptype_columns()

    print('Wrapping positions...')
    t1 = perf_counter()
    wrap_positions(data_manager)
    t2 = perf_counter()
    print(f'Done in {t2-t1:.2f} seconds.')

    print('Running FOF6D...')
    t1 = perf_counter()
    run_fof6d(data_manager, nproc=config.get('nproc', 1))
    t2 = perf_counter()
    print(f'Done in {t2-t1:.2f} seconds.')

  print('Calculating group properties...')
  t1 = perf_counter()
  data_manager.initialise_group_data()
  calculate_group_properties(data_manager)
  t2 = perf_counter()
  print(f'Done in {t2-t1:.2f} seconds.')

  print('Assigning particle lists...')
  t1 = perf_counter()
  get_particle_lists(data_manager)
  t2 = perf_counter()
  print(f'Done in {t2-t1:.2f} seconds.')

    print('Saving datasets...')
    t1 = perf_counter()
    save_group_properties(data_manager, outfile)
    t2 = perf_counter()
    print(f'Done in {t2-t1:.2f} seconds.')