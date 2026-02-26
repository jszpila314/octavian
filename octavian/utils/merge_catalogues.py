import h5py
import numpy as np
import warnings
from yaml import safe_load

warnings.filterwarnings("ignore", category=RuntimeWarning)

def merge_catalogues(files: list[str], outfile: str, configfile: str) -> None:
  with open(configfile, 'r') as f:
    config = safe_load(f)

  galaxy_parent_halo = []
  halo_masses = []
  galaxy_masses = []
  file_lengths = {'halos': {}, 'galaxies': {}}

  for file in files:
    with h5py.File(file, 'r') as f:
      file_lengths['halos'][file] = len(f['halo_data']['dicts/masses.total'])
      try:
        file_lengths['galaxies'][file] = len(f['galaxy_data']['dicts/masses.total'])
      except:
        file_lengths['galaxies'][file] = 0

      halo_masses.append(f['halo_data']['dicts/masses.total'][:])
      try:
        file_galaxy_masses = f['galaxy_data']['dicts/masses.stellar'][:]
        if len(file_galaxy_masses) != 0:
          galaxy_parent_halo.append(f['galaxy_data']['parent_halo_index'][:] + len(halo_masses))
          galaxy_masses.append(file_galaxy_masses)
      except: pass
  print(file_lengths)
  halo_masses = np.concatenate(halo_masses)
  galaxy_masses = np.concatenate(galaxy_masses)
  galaxy_parent_halo = np.concatenate(galaxy_parent_halo)

  halo_order = np.argsort(halo_masses)
  galaxy_order = np.argsort(galaxy_masses)
  galaxy_parent_halo = halo_order[galaxy_parent_halo]

  with h5py.File(outfile, 'w') as f_out:
    halo_group = f_out.create_group('halo_data')
    galaxy_group = f_out.create_group('galaxy_data')

    for dataset in config['dataset_columns'].keys():
      print(dataset)
      if 'list' in dataset or 'groupID' in dataset or 'parent_halo_index' in dataset: continue
      halo_data = []
      galaxy_data = []
      for file in files:
        with h5py.File(file, 'r') as f:

          try: halo_data.append(f['halo_data'][dataset][:])
          except:
            if '_L' in dataset: halo_data.append(np.full((file_lengths['halos'][file], 3), np.nan))
            else: halo_data.append(np.full(file_lengths['halos'][file], np.nan))

          if file_lengths['galaxies'][file] == 0: continue
          try: galaxy_data.append(f['galaxy_data'][dataset][:])
          except:
            if '_L' in dataset: galaxy_data.append(np.full((file_lengths['galaxies'][file], 3), np.nan))
            else: galaxy_data.append(np.full(file_lengths['galaxies'][file], np.nan))

      halo_group[dataset] = np.concatenate(halo_data)[halo_order]
      galaxy_group[dataset] = np.concatenate(galaxy_data)[galaxy_order]

    halo_group['HaloID'] = np.arange(np.sum(list(file_lengths['halos'].values())))
    galaxy_group['GalID'] = np.arange(np.sum(list(file_lengths['galaxies'].values())))
    galaxy_group['parent_halo_index'] = galaxy_parent_halo
    


