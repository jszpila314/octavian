from time import perf_counter

from octavian.data_manager import DataManager, save_group_properties
from octavian.utils import wrap_positions
from octavian.halo_finder import run_fof6d
from octavian.group_properties_calc import calculate_group_properties, get_particle_lists

from yaml import safe_load

def run(snapshot: str, outfile: str, configfile: str, logfile: str, comm=None):
  with open(configfile, 'r') as f:
    config = safe_load(f)

  config['Tlim'] = float(config['Tlim'])

  data_manager = DataManager(snapshot, logfile, config, comm=comm)

  wrap_positions(data_manager)
  
  run_fof6d(data_manager, nproc=config.get('nproc', 1))

  data_manager.initialise_group_data()
  calculate_group_properties(data_manager)
  
  get_particle_lists(data_manager)
  
  save_group_properties(data_manager, outfile)
  