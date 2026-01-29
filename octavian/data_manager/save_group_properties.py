from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from octavian.data_manager import DataManager

import h5py
import os
import numpy as np
import warnings


warnings.filterwarnings("ignore", category=RuntimeWarning)

def save_group_properties(data_manager: DataManager, filename: str) -> None:
  config = data_manager.config

  if os.path.exists(filename):
    os.remove(filename)
  
  with h5py.File(filename, 'w') as f:
    halo_data = f.create_group('halo_data')
    galaxy_data = f.create_group('galaxy_data')

    halo_columns = data_manager.group_data['halos'].columns
    galaxy_columns = data_manager.group_data['galaxies'].columns

    for dataset_name, column in config['dataset_columns'].items():
      if np.all(np.isin(column, halo_columns)):
        halo_data.create_dataset(dataset_name, data=data_manager.group_data['halos'][column].to_numpy(), compression=1)
      if np.all(np.isin(column, galaxy_columns)):
        galaxy_data.create_dataset(dataset_name, data=data_manager.group_data['galaxies'][column].to_numpy(), compression=1)
