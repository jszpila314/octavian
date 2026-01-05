from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from octavian.data_manager import DataManager


import h5py
import os
import warnings



warnings.filterwarnings("ignore", category=RuntimeWarning)

def save_group_properties(data_manager: DataManager, filename: str) -> None:
  config = data_manager.config

  column_to_dataset_map = {}

  # position
  column_to_dataset_map['pos'] = 'pos'
  column_to_dataset_map['minpotpos'] = 'minpotpos'

  # velocity
  column_to_dataset_map['vel'] = 'vel'
  column_to_dataset_map['minpotvel'] = 'minpotvel'

  # npart
  for group in ['gas', 'star', 'dm', 'bh']:
    column_to_dataset_map[f'n{group}'] = f'n{group}'

  # masses
  for group in ['gas', 'star', 'dm', 'bh', 'HI', 'H2', 'total']:
    group_name = 'stellar' if group == 'star' else group
    column_to_dataset_map[f'mass_{group}'] = f'dicts/masses.{group_name}'
    column_to_dataset_map[f'mass_{group}_30kpc'] = f'dicts/masses.{group_name}_30kpc'

  for group in ['gas', 'star', 'dm', 'bh', 'baryon', 'total']:
    group_name = 'stellar' if group == 'star' else group

    # radii
    for radius_size in ['r20', 'half_mass', 'r80']:
      column_to_dataset_map[f'radius_{group}_{radius_size}'] = f'dicts/radii.{group_name}_{radius_size}'

    # velocity dispersions
    column_to_dataset_map[f'velocity_dispersion_{group}'] = f'dicts/velocity_dispersions.{group_name}'

    # rotation
    column_to_dataset_map[f'L_{group}'] = f'dicts/rotation.{group_name}_L'
    for property in ['ALPHA', 'BETA', 'BoverT', 'kappa_rot']:
      column_to_dataset_map[f'{property}_{group}'] = f'dicts/rotation.{group_name}_{property}'

  # virial quantities
  for property in ['r200', 'circular_velocity', 'spin_param', 'temperature']:
    column_to_dataset_map[property] = f'dicts/virial_quantities.{property}'

  for factor in ['200', '500', '2500']:
    column_to_dataset_map[f'radius_{factor}_c'] = f'dicts/virial_quantities.r{factor}c'
    column_to_dataset_map[f'mass_{factor}_c'] = f'dicts/virial_quantities.m{factor}c'

  # gas masses
  for group in ['HI', 'H2']:
    column_to_dataset_map[f'mass_{group}'] = f'dicts/masses.{group}'

  # metallicties
  for property in ['mass_weighted', 'sfr_weighted', 'mass_weighted_cgm', 'temp_weighted_cgm', 'stellar']:
    column_to_dataset_map[f'metallicity_{property}'] = f'dicts/metallicities.{property}'

  # ages
  for property in ['mass_weighted', 'metal_weighted']:
    column_to_dataset_map[f'age_{property}'] = f'dicts/ages.{property}'

  # temperatures
  for property in ['mass_weighted', 'mass_weighted_cgm', 'metal_weighted_cgm']:
    column_to_dataset_map[f'temp_{property}'] = f'dicts/temperatures.{property}'

  # local densities
  for property in ['local_mass_density', 'local_number_density']:
    for radius in ['300', '1000', '3000']:
      column_to_dataset_map[f'{property}_{radius}'] = f'dicts/{property}.{radius}'


  if os.path.exists(filename):
    os.remove(filename)
  
  with h5py.File(filename, 'w') as f:
    halo_data = f.create_group('halo_data')
    galaxy_data = f.create_group('galaxy_data')

    for group in config['groups']:
      for column, dataset_name in column_to_dataset_map.items():
        if column in ['minpotpos', 'minpotvel'] and group == 'galaxies': continue
        if 'virial' in dataset_name and group == 'galaxies': continue
        if'_30kpc' in dataset_name and group == 'halos': continue

        if column == 'pos': column = ['x_total', 'y_total', 'z_total']
        if column == 'vel': column = ['vx_total', 'vy_total', 'vz_total']
        if column == 'minpotpos': column = ['minpot_x', 'minpot_y', 'minpot_z']
        if column == 'minpotvel': column = ['minpot_vx', 'minpot_vy', 'minpot_vz']

        if 'L_' in column:
          ptype = column[2:]
          column = [f'Lx_{ptype}', f'Ly_{ptype}', f'Lz_{ptype}']
        
        if group == 'halos':
          halo_data.create_dataset(dataset_name, data=data_manager.group_data[group][column].to_numpy(), compression=1)
        else:
          galaxy_data.create_dataset(dataset_name, data=data_manager.group_data[group][column].to_numpy(), compression=1)
