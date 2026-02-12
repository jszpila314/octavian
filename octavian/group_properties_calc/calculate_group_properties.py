from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from octavian.data_manager import DataManager

import numpy as np
import pandas as pd
import unyt
from sklearn.neighbors import NearestNeighbors
from functools import partial
from astropy import constants as const

# Suppress pandas fragmented frame performance warnings (superfluous)  https://stackoverflow.com/a/76306267
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


def broadcast_properties(data: pd.DataFrame, group_data: pd.DataFrame, groupID: str, properties: list[str] | str) -> np.ndarray:
  return data[[groupID]].merge(group_data[properties], left_on=groupID, right_index=True)[properties].to_numpy()


def calculateGroupProperties_common(data_manager: DataManager, group_name: str, particle_type: str) -> None:
  config = data_manager.config

  group_props_columns = ['HaloID', 'GalID', 'ptype', 'mass', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'potential']
  if particle_type == 'total':
    data = pd.concat([data_manager.data[ptype][group_props_columns] for ptype in config['ptypes']], ignore_index=True)
  elif particle_type == 'baryon':
    data = pd.concat([data_manager.data[ptype][group_props_columns] for ptype in config['ptypes_baryon']], ignore_index=True)
  else:
    data = data_manager.data[particle_type][group_props_columns].copy()

  if group_name == 'galaxies':
    data = data.loc[data['GalID'] != -1]


  group_data = data_manager.group_data[group_name]
  groupID = config['groupIDs'][group_name]

  data_grouped = data.groupby(by=groupID, observed=True)

  # nparticles
  group_data[f'n{particle_type}'] = data_grouped.size()


  # group masses
  if particle_type == 'bh':
    group_data[f'mass_{particle_type}'] = data_grouped['mass'].max()
  else:
    group_data[f'mass_{particle_type}'] = data_grouped['mass'].sum()


  # minpotpos, mipotvel
  if group_name == 'halos' and particle_type == 'total':
    minimum_potential_index = data_grouped['potential'].idxmin()
    group_data[['minpot_x', 'minpot_y', 'minpot_z', 'minpot_vx', 'minpot_vy', 'minpot_vz']] = data.loc[minimum_potential_index, ['HaloID', 'x', 'y', 'z', 'vx', 'vy', 'vz']].set_index('HaloID')


  # centre of mass, com velocity
  for column in ['x', 'y', 'z', 'vx', 'vy', 'vz']:
    data['temp'] = data.eval(f'{column} * mass')
    group_data[f'{column}_{particle_type}'] = data_grouped['temp'].sum() / group_data[f'mass_{particle_type}']

  data.drop(columns='temp', inplace=True)


  # velocity dispersion
  group_velocity_columns = [f'vx_{particle_type}', f'vy_{particle_type}', f'vz_{particle_type}']
  data[['rel_vx', 'rel_vy', 'rel_vz']] = data[['vx', 'vy', 'vz']] - broadcast_properties(data, group_data, groupID, group_velocity_columns)

  data[f'velocity_dispersion'] = np.sum(data[['rel_vx', 'rel_vy', 'rel_vz']]**2, axis=1)
  group_data[f'velocity_dispersion_{particle_type}'] = np.sqrt(data_grouped['velocity_dispersion'].sum() / group_data[f'n{particle_type}'])

  data.drop(columns=['rel_vx', 'rel_vy', 'rel_vz', 'velocity_dispersion'], inplace=True)


  # radius
  group_position_columns = ['minpot_x', 'minpot_y', 'minpot_z'] if group_name == 'halos' else [f'x_{particle_type}', f'y_{particle_type}', f'z_{particle_type}']
                                                                                               
  data[['rel_x', 'rel_y', 'rel_z']] = data[['x', 'y', 'z']] - broadcast_properties(data, group_data, groupID, group_position_columns)
  data.drop(columns=['x', 'y', 'z'], inplace=True)
  data['radius'] = np.linalg.norm(data[['rel_x', 'rel_y', 'rel_z']], axis=1)


  # angular momentum
  group_velocity_columns = ['minpot_vx', 'minpot_vy', 'minpot_vz'] if group_name == 'halos' else [f'vx_{particle_type}', f'vy_{particle_type}', f'vz_{particle_type}']

  data[['rel_vx', 'rel_vy', 'rel_vz']] = data[['vx', 'vy', 'vz']] - broadcast_properties(data, group_data, groupID, group_velocity_columns)
  data['ktot'] = data.eval('0.5 * mass * ((rel_vx**2) + (rel_vy**2) + (rel_vz**2))')
  data.drop(columns=['vx', 'vy', 'vz'], inplace=True)

  data[['rel_px', 'rel_py', 'rel_pz']] = data[['rel_vx', 'rel_vy', 'rel_vz']].multiply(data['mass'], axis='index')
  data.drop(columns=['rel_vx', 'rel_vy', 'rel_vz'], inplace=True)

  data[['Lx', 'Ly', 'Lz']] = np.cross(data[['rel_x', 'rel_y', 'rel_z']], data[['rel_px', 'rel_py', 'rel_pz']])
  for direction in ['x', 'y', 'z']:
    group_data[f'L{direction}_{particle_type}'] = data_grouped[f'L{direction}'].sum()
  data.drop(columns=['rel_px', 'rel_py', 'rel_pz'], inplace=True)

  angular_momentum_columns = [f'Lx_{particle_type}', f'Ly_{particle_type}', f'Lz_{particle_type}']
  data[['Lx_group', 'Ly_group', 'Lz_group']] = broadcast_properties(data, group_data, groupID, angular_momentum_columns)
  data['L_dot_L_group'] = data.eval('Lx * Lx_group + Ly * Ly_group + Lz * Lz_group')
  data.drop(columns=['Lx', 'Ly', 'Lz'], inplace=True)

  group_data[f'L_{particle_type}'] = np.linalg.norm(group_data[[f'Lx_{particle_type}', f'Ly_{particle_type}', f'Lz_{particle_type}']], axis=1)
  group_data[f'ALPHA_{particle_type}'] = np.arctan2(group_data[f'Ly_{particle_type}'], group_data[f'Lz_{particle_type}'])
  group_data[f'BETA_{particle_type}'] = np.arcsin(group_data[f'Lx_{particle_type}'] / group_data[f'L_{particle_type}'])

  group_data[f'BoverT_{particle_type}'] = 2 * data.loc[data['L_dot_L_group'] < 0].groupby(by='HaloID', observed=True)['mass'].sum() / group_data[f'mass_{particle_type}']


  # rotation quantities
  data['rz'] = data.eval('sqrt((rel_y * Lz_group - rel_z * Ly_group)**2 + (rel_z * Lx_group - rel_x * Lz_group)**2 + (rel_x * Ly_group - rel_y * Lx_group)**2)')
  data.drop(columns=['rel_x', 'rel_y', 'rel_z', 'Lx_group', 'Ly_group', 'Lz_group'], inplace=True)
  
  data['krot'] = data.eval('(0.5 * (L_dot_L_group / rz)**2) * mass**(-1)') # for some reason, '/ mass' breaks due to forbidden control characters, no clue

  ordered_rotation_grouped = data.loc[data['rz'] > 0, ['HaloID', 'krot', 'ktot']].groupby(by='HaloID', observed=True)
  group_data[f'kappa_rot_{particle_type}'] = ordered_rotation_grouped['krot'].sum() / ordered_rotation_grouped['ktot'].sum()

  data.drop(columns=['L_dot_L_group', 'rz', 'krot', 'ktot'], inplace=True)

  angular_quantities = [f'velocity_dispersion_{particle_type}', f'Lx_{particle_type}', f'Ly_{particle_type}', f'Lz_{particle_type}', f'BoverT_{particle_type}', f'kappa_rot_{particle_type}']
  group_data.loc[group_data[f'n{particle_type}'] < 3, angular_quantities] = 0.


  # radial quantities
  data.sort_values(by='radius', inplace=True)
  data_grouped = data.groupby(by='HaloID', observed=True)

  data['cumulative_mass'] = data_grouped['mass'].cumsum()
  data['cumulative_mass_fraction'] = data['cumulative_mass'] / broadcast_properties(data, group_data, groupID, f'mass_{particle_type}')

  for quantile, col_name in zip([0.2, 0.5, 0.8], ['r20', 'half_mass', 'r80']):
    data.loc[data['cumulative_mass_fraction'] < quantile, 'cumulative_mass_fraction'] = np.nan
    minimum_cummass_index = data_grouped['cumulative_mass_fraction'].idxmin(skipna=True)
    group_data[f'radius_{particle_type}_{col_name}'] = data.loc[minimum_cummass_index, [groupID, 'radius']].set_index(groupID)


  # virial quantities -> around minpotpos
  if group_name == 'halos' and particle_type == 'total':
    group_data['r200'] = data_manager.simulation['r200_factor'] * (group_data[f'mass_{particle_type}'])**(1/3)
    group_data['circular_velocity'] = np.sqrt(unyt.G.to('(km**2 * kpc)/(Msun * s**2)').d * group_data[f'mass_{particle_type}'] / group_data['r200'])
    group_data['temperature'] = 3.6e5 * (group_data['circular_velocity'] / 100.0)**2
    group_data['spin_param'] = group_data[f'L_{particle_type}'] / (np.sqrt(2) * group_data[f'mass_{particle_type}'] * group_data['circular_velocity'] * group_data['r200'])

    volume_factor = 4./3.*np.pi
    rhocrit = (data_manager.simulation['rhocrit'] * data_manager.create_unit_quantity('rhocrit')).to('Msun / (kpc*a)**3').d
    data['overdensity'] = data['cumulative_mass'] / (volume_factor * data['radius']**3) / rhocrit

    for factor in [200, 500, 2500]:
      data.loc[data['overdensity'] < factor, ['radius', 'cumulative_mass']] = np.nan
      group_data[f'radius_{factor}_c'] = data_grouped['radius'].last()
      group_data[f'mass_{factor}_c'] = data_grouped['cumulative_mass'].last()


def calculateGroupProperties_gas(data_manager: DataManager, group_name: str) -> None:
  config = data_manager.config

  data = data_manager.data['gas'].copy()

  if group_name == 'galaxies':
    data = data.loc[data['GalID'] != -1]

  group_data = data_manager.group_data[group_name]

  config = data_manager.config
  groupID = config['groupIDs'][group_name]

  data_grouped = data.groupby(by=groupID, observed=True)



  # remember HI, H2 for aperture masses
  if group_name == 'halos':
    data['mass_HI'] = data['mass_HI']
    data_manager.data['gas']['mass_H2'] = data['mass_H2']

  group_data['mass_HI'] = data_grouped['mass_HI'].sum()
  group_data['mass_H2'] = data_grouped['mass_H2'].sum()

  data.drop(columns=['nh', 'fHI', 'fH2', 'mass_HI', 'mass_H2'], inplace=True)
  

  # sfr
  group_data['sfr'] = data_grouped['sfr'].sum()


  # metallicity
  data['metallicity_mass_weighted'] = data.eval('metallicity * mass')
  data['metallicity_sfr_weighted'] = data.eval('metallicity * sfr')

  group_data['metallicity_mass_weighted'] = data_grouped['metallicity_mass_weighted'].sum() / group_data['mass_gas']
  group_data['metallicity_sfr_weighted'] = data_grouped['metallicity_sfr_weighted'].sum() / group_data['sfr']


  # cgm mass, temperatures, metallicity
  data['temp_mass_weighted'] = data.eval('temperature * mass')
  data['temp_metal_weighted'] = data.eval('temp_mass_weighted * metallicity')

  cgm_mask = data['rho'] < config['nHlim']
  data_cgm_grouped = data.loc[cgm_mask, ['HaloID', 'mass', 'temp_mass_weighted', 'temp_metal_weighted', 'metallicity_mass_weighted', 'metallicity_sfr_weighted']].groupby(by='HaloID', observed=True)
  group_data['mass_cgm'] = data_cgm_grouped['mass'].sum()

  group_data['temp_mass_weighted'] = data_grouped['temp_mass_weighted'].sum() / group_data['mass_gas']
  group_data['temp_mass_weighted_cgm'] = data_cgm_grouped['temp_mass_weighted'].sum()

  group_data['temp_metal_weighted_cgm'] = data_cgm_grouped['temp_metal_weighted'].sum() / group_data['temp_mass_weighted_cgm']
  group_data['temp_mass_weighted_cgm'] /= group_data['mass_cgm']

  data.drop(columns=['temp_mass_weighted'], inplace=True)

  group_data['metallicity_mass_weighted_cgm'] = data_cgm_grouped['metallicity_mass_weighted'].sum()
  group_data['metallicity_temp_weighted_cgm'] = data_cgm_grouped['temp_metal_weighted'].sum() / group_data['metallicity_mass_weighted_cgm']
  group_data['metallicity_mass_weighted_cgm'] /= group_data['mass_cgm']


def calculateGroupProperties_star(data_manager: DataManager, group_name: str) -> None:
  data = data_manager.data['star'].copy()

  if group_name == 'galaxies':
    data = data.loc[data['GalID'] != -1]

  group_data = data_manager.group_data[group_name]

  config = data_manager.config
  groupID = config['groupIDs'][group_name]

  data_grouped = data.groupby(by=groupID, observed=True)


  # metallicity
  data['metallicity_stellar'] = data.eval('metallicity * mass')

  group_data['metallicity_stellar'] = data_grouped['metallicity_stellar'].sum()


  # age
  data['age_mass_weighted'] = data['age'] * data['mass']
  data['age_metal_weighted'] = data['age'] * data['mass'] * data['metallicity']

  group_data['age_mass_weighted'] = data_grouped['age_mass_weighted'].sum() / group_data['mass_star']
  group_data['age_metal_weighted'] = data_grouped['age_metal_weighted'].sum() / group_data['metallicity_stellar']

  group_data['metallicity_stellar'] /= data_grouped['mass'].sum()


def calculateGroupProperties_bh(data_manager: DataManager, group_name: str) -> None:
  data = data_manager.data['bh']

  if group_name == 'galaxies':
    data = data.loc[data['GalID'] != -1]

  group_data = data_manager.group_data[group_name]

  config = data_manager.config
  groupID = config['groupIDs'][group_name]

  data_grouped = data.groupby(by=groupID, observed=True)
  max_mass_index = data_grouped['mass'].idxmax()
  data = data.loc[max_mass_index].set_index(groupID)


  # bhmdot
  group_data['bhmdot'] = data['bhmdot'].copy()


  # bh_fedd
  FRAD = 0.1  # assume 10% radiative efficiency
  edd_factor = (4 * np.pi * const.G * const.m_p / (FRAD * const.c * const.sigma_T)).to('1/yr').value
  group_data['bh_fedd'] = data['bhmdot'] / (edd_factor * data['mass'])


def calculate_aperture_masses(halo: pd.DataFrame, aperture: float, galaxy_positions: np.ndarray) -> pd.DataFrame:
  galaxy_ids = halo['GalID'].unique()
  galaxy_ids = galaxy_ids[galaxy_ids != -1]
  if len(galaxy_ids) == 0: return pd.DataFrame()

  data = []
  for GalID in galaxy_ids:
    relative_positions = halo[['x', 'y', 'z']] - galaxy_positions[GalID]
    halo['radius'] = np.linalg.norm(relative_positions, axis=1)

    masses = halo.loc[halo['radius'] < aperture].groupby(by='ptype')['mass'].sum()
    masses.name = GalID
    data.append(masses)

  return pd.DataFrame(data=data)


def calculate_local_densities(data_manager: DataManager) -> None:

  config = data_manager.config
  groups = config['groups']

  for group in groups:

    # safeguard in case a group is not filled
    if len(group_data) == 0:
      print(f"No group data!")
      continue

    group_data = data_manager.group_data[group]
    pos = group_data[['x_total', 'y_total', 'z_total']].to_numpy()
    mass = group_data['mass_total'].to_numpy()

    neighbors = NearestNeighbors()
    neighbors.fit(pos)

    for radius in [300., 1000., 3000.]:
      volume = 4./3. * np.pi * radius**3

      df = pd.DataFrame({
        'indexes': neighbors.radius_neighbors(pos, radius=radius, return_distance=False)
      })

      df = df.explode('indexes').dropna()

      df['mass'] = mass[df['indexes'].astype('int')]
      grouped = df.groupby(level=0)

      group_data[f'local_mass_density_{int(radius)}'] = grouped['mass'].sum() / volume
      group_data[f'local_number_density_{int(radius)}'] = grouped.size() / volume


def calculate_group_properties(data_manager: DataManager) -> None:
  config = data_manager.config
  for ptype in config['ptypes']:
    data_manager.load_property('potential', ptype)

  groups = config['groups']

  group_props_columns = ['HaloID', 'GalID', 'ptype', 'mass', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'potential']
  columns_to_drop = ['vx', 'vy', 'vz', 'potential']
  to_process = config['to_process']

  # total common
  if 'total' in to_process:
    for group in groups:
      calculateGroupProperties_common(data_manager, group, 'total')


  # dm common
  if 'dm' in to_process:
    for group in groups:
      calculateGroupProperties_common(data_manager, group, 'dm')

    data_manager.data['dm'].drop(columns=columns_to_drop, inplace=True)


  # baryon common
  if 'baryon' in to_process:
    for group in groups:
      calculateGroupProperties_common(data_manager, group, 'baryon')


  # gas common
  if 'gas' in to_process:
    for group in groups:
      calculateGroupProperties_common(data_manager, group, 'gas')
  
    data_manager.data['gas'].drop(columns=columns_to_drop, inplace=True)


  # star common
  if 'star' in to_process:
    for group in groups:
      calculateGroupProperties_common(data_manager, group, 'star')

    data_manager.data['star'].drop(columns=columns_to_drop, inplace=True)


  # bh common
  if 'bh' in to_process:
    for group in groups:
      calculateGroupProperties_common(data_manager, group, 'bh')

    data_manager.data['bh'].drop(columns=columns_to_drop, inplace=True)


  # gas
  if 'gas' in to_process:
    for property in ['rho', 'nh', 'fH2', 'metallicity', 'sfr', 'temperature']:
      data_manager.load_property(property, 'gas')

    # gas massses
    data = data_manager.data['gas']
    data['fHI'] = data.eval('nh / mass')
    not_conserving_mass = data.eval('(fHI + fH2) > 1')
    data.loc[not_conserving_mass, 'fHI'] = 1. - data.loc[not_conserving_mass, 'fH2']

    data['mass_HI'] = config['XH'] * data['fHI'] * data['mass']
    data['mass_H2'] = config['XH'] * data['fH2'] * data['mass']

    for group in groups:
      calculateGroupProperties_gas(data_manager, group)


  # star
  if 'star' in to_process:
    for property in ['age', 'metallicity']:
      data_manager.load_property(property, 'star')

    for group in groups:
      calculateGroupProperties_star(data_manager, group)



  # bh
  if 'bh' in to_process:
    for property in ['bhmdot']:
      data_manager.load_property(property, 'bh')
    
    star_props_columns = ['HaloID', 'GalID', 'ptype', 'mass', 'bhmdot']
    for group in groups:
      calculateGroupProperties_bh(data_manager, group)

  # aperture
  if 'apertures' in to_process:
    aperture_props_columns = ['HaloID', 'GalID', 'ptype', 'mass', 'x',  'y', 'z']
    data = pd.concat([data_manager.data[ptype][aperture_props_columns] for ptype in config['ptypes']])

    aperture_HI_columns = ['HaloID', 'GalID', 'ptype', 'mass_HI', 'x',  'y', 'z']
    HI_gas = data_manager.data['gas'][aperture_HI_columns].copy()
    HI_gas.rename(columns={'mass_HI': 'mass'}, inplace=True)
    HI_gas['ptype'] = 'HI'

    aperture_H2_columns = ['HaloID', 'GalID', 'ptype', 'mass_H2', 'x',  'y', 'z']
    H2_gas = data_manager.data['gas'][aperture_H2_columns].copy()
    H2_gas.rename(columns={'mass_H2': 'mass'}, inplace=True)
    H2_gas['ptype'] = 'H2'

    data = pd.concat([data, HI_gas, H2_gas], ignore_index=True)

    aperture = 30.
    galaxy_positions = data_manager.group_data['galaxies'][['x_total', 'y_total', 'z_total']].to_numpy()

    process_halo = partial(calculate_aperture_masses, aperture=aperture, galaxy_positions=galaxy_positions)
    aperture_masses = data.groupby(by='HaloID').apply(process_halo, include_groups = False).reset_index(names=['HaloID', 'GalID'])
    aperture_masses.set_index('GalID', inplace=True)

    # when we filter the snapshot not all particle types are necessarily present
    for ptype in ['gas', 'dm', 'star', 'bh', 'HI', 'H2']:
      col_name = f'mass_{ptype}_30kpc'
    if ptype in aperture_masses.columns:
        data_manager.group_data['galaxies'][col_name] = aperture_masses[ptype]
    else:
        data_manager.group_data['galaxies'][col_name] = 0.0 # set equal to 0 if it does not exist

    # then we only sum from existing columns to avoid the key error
    mass_cols = [col for col in ['mass_gas_30kpc', 'mass_dm_30kpc', 'mass_star_30kpc', 'mass_bh_30kpc'] 
                if col in data_manager.group_data['galaxies'].columns]

    if mass_cols: # check these exist
        data_manager.group_data['galaxies']['mass_total_30kpc'] = data_manager.group_data['galaxies'][mass_cols].sum(axis=1)
    else:
        data_manager.group_data['galaxies']['mass_total_30kpc'] = 0.0

  if 'local_densities' in to_process:
    calculate_local_densities(data_manager)
