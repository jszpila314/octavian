from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from octavian.data_manager import DataManager

import h5py
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def merge_catalogues(files: list[str], outfile: str) -> None:
  galaxy_parent_halo = []
  halo_masses = []
  galaxy_masses = []

  halo_datasets = []
  galaxy_datasets = []
  for file in files:
    with h5py.File(file, 'r') as f:
      halo_datasets.append(list(f['halo_data'].keys()))
      galaxy_datasets.append(list(f['galaxy_data'].keys()))
      galaxy_parent_halo.append(f['galaxy_data']['parent_halo_index'][:] + len(halo_masses))
      halo_masses.append(f['halo_data']['dicts/masses.total'][:])
      galaxy_masses.append(f['galaxy_data']['dicts/masses.stellar'][:])
  
  halo_datasets = np.unique(np.concatenate(halo_datasets))
  galaxy_datasets = np.unique(np.concatenate(galaxy_datasets))

  halo_masses = np.concatenate(halo_masses)
  galaxy_masses = np.concatenate(galaxy_masses)
  galaxy_parent_halo = np.concatenate(galaxy_parent_halo)

  halo_order = np.argsort(halo_masses)
  galaxy_order = np.argsort(galaxy_masses)
  galaxy_parent_halo = halo_order[galaxy_parent_halo]

  with h5py.File(outfile, 'r') as f_out:
    for dataset in halo_datasets:
      data = []
      for file in files:
        with h5py.File(file, 'r') as f:
          try: data.append(f['halo_data'][dataset][:])
          except: pass
      f_out['halo_data'][dataset] = np.concatenate(data)[halo_order]

    f_out['halo_data']['HaloID'] = np.arange(len(data))

    for dataset in galaxy_datasets:
      data = []
      for file in files:
        with h5py.File(file, 'r') as f:
          try: data.append(f['galaxy_data'][dataset][:])
          except: pass
      f_out['galaxy_data'][dataset] = np.concatenate(data)[galaxy_order]

    f_out['galaxy_data']['GalID'] = np.arange(len(data))
    f_out['galaxy_data']['parent_halo_index'] = galaxy_parent_halo
    


