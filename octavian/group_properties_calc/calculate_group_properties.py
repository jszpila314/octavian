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
from time import perf_counter

from scipy.spatial import KDTree

from octavian.group_properties_calc.group_computations import (
    compute_angular_momentum,
    compute_aperture_component_properties,
    compute_central_galaxy_flags,
    compute_gas_scalar_sums,
    compute_rotation_quantities,
    compute_radial_quantiles_and_rmax,
    compute_star_scalar_sums,
    compute_virial_quantities,
)

from octavian.group_properties_calc.group_helpers import (
    sum_per_group,
    count_per_group,
    max_value_per_group,
    max_idx_per_group,
    min_idx_per_group,
    broadcast_to_particles,
    sort_by_group,
    weighted_mean_per_group,
    extract_particle_arrays,
)

# Suppress pandas fragmented frame performance warnings (superfluous)  https://stackoverflow.com/a/76306267
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


def _safe_divide(numerator, denominator, fill=np.nan):
  numerator = np.asarray(numerator, dtype=float)
  denominator = np.asarray(denominator, dtype=float)
  result = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), fill, dtype=float)
  return np.divide(numerator, denominator, out=result, where=denominator != 0)


def _ahf_lineage_for_ids(tree, source_halo_ids, output_halo_ids):
  source_halo_ids = np.asarray(source_halo_ids, dtype=np.int64)
  output_halo_ids = np.asarray(output_halo_ids, dtype=np.int64)
  n = len(source_halo_ids)

  ahf_halo_id = np.full(n, -1, dtype=np.int64)
  ahf_parent_halo_id = np.full(n, -1, dtype=np.int64)
  ahf_top_halo_id = np.full(n, -1, dtype=np.int64)
  ahf_depth = np.full(n, -1, dtype=np.int64)
  caesar_parent_halo_index = np.full(n, -1, dtype=np.int64)
  caesar_top_halo_index = np.full(n, -1, dtype=np.int64)

  if len(tree.halo_ids) == 0:
    return (
        ahf_halo_id,
        ahf_parent_halo_id,
        ahf_top_halo_id,
        ahf_depth,
        caesar_parent_halo_index,
        caesar_top_halo_index,
    )

  max_halo_id = len(tree._id_to_idx)
  valid = (source_halo_ids >= 0) & (source_halo_ids < max_halo_id)
  valid[valid] = tree._id_to_idx[source_halo_ids[valid]] != -1
  valid_ids = source_halo_ids[valid]
  tree_rows = tree._id_to_idx[valid_ids]

  original_ids = tree.halo_ids.copy()
  if tree.properties is not None and 'ID' in tree.properties:
    original_ids = tree.properties['ID'].to_numpy(dtype=np.int64)

  original_by_halo_id = np.full(max_halo_id, -1, dtype=np.int64)
  original_by_halo_id[tree.halo_ids] = original_ids

  output_by_halo_id = np.full(max_halo_id, -1, dtype=np.int64)
  present_output = (output_halo_ids >= 0) & (output_halo_ids < max_halo_id)
  output_by_halo_id[output_halo_ids[present_output]] = output_halo_ids[present_output]

  parent_ids = np.full(n, -1, dtype=np.int64)
  parent_ids[valid] = tree.parent_ids[tree_rows]
  top_ids = np.full(n, -1, dtype=np.int64)
  top_ids[valid] = tree.field_map[valid_ids]

  ahf_halo_id[valid] = original_by_halo_id[valid_ids]
  parent_valid = valid & (parent_ids >= 0) & (parent_ids < max_halo_id)
  ahf_parent_halo_id[parent_valid] = original_by_halo_id[parent_ids[parent_valid]]
  top_valid = valid & (top_ids >= 0) & (top_ids < max_halo_id)
  ahf_top_halo_id[top_valid] = original_by_halo_id[top_ids[top_valid]]
  ahf_depth[valid] = tree.depths[tree_rows]

  caesar_parent_halo_index[parent_valid] = output_by_halo_id[parent_ids[parent_valid]]
  caesar_top_halo_index[top_valid] = output_by_halo_id[top_ids[top_valid]]

  return (
      ahf_halo_id,
      ahf_parent_halo_id,
      ahf_top_halo_id,
      ahf_depth,
      caesar_parent_halo_index,
      caesar_top_halo_index,
  )


def _ahf_ancestor_chains(tree, source_halo_ids):
  source_halo_ids = np.asarray(source_halo_ids, dtype=np.int64)
  if len(tree.halo_ids) == 0:
    return [np.empty(0, dtype=np.int64) for _ in source_halo_ids]

  max_halo_id = len(tree._id_to_idx)
  original_ids = tree.halo_ids.copy()
  if tree.properties is not None and 'ID' in tree.properties:
    original_ids = tree.properties['ID'].to_numpy(dtype=np.int64)

  original_by_halo_id = np.full(max_halo_id, -1, dtype=np.int64)
  original_by_halo_id[tree.halo_ids] = original_ids

  chains = []
  for source_halo_id in source_halo_ids:
    if source_halo_id < 0 or source_halo_id >= max_halo_id:
      chains.append(np.empty(0, dtype=np.int64))
      continue

    row = tree._id_to_idx[source_halo_id]
    if row == -1:
      chains.append(np.empty(0, dtype=np.int64))
      continue

    chain = []
    seen = set()
    parent = tree.parent_ids[row]
    while parent != -1 and parent not in seen and 0 <= parent < max_halo_id:
      parent_row = tree._id_to_idx[parent]
      if parent_row == -1:
        break
      chain.append(original_by_halo_id[parent])
      seen.add(parent)
      parent = tree.parent_ids[parent_row]

    chains.append(np.asarray(chain, dtype=np.int64))

  return chains


def _assign_halo_source_properties(data_manager: DataManager) -> None:
  if data_manager.config.get('halo_source') != 'ahf':
    return
  if 'halos' not in data_manager.group_data or not hasattr(data_manager, 'halo_tree'):
    return

  tree = data_manager.halo_tree
  halo_data = data_manager.group_data['halos']
  halo_ids = halo_data.index.to_numpy(dtype=np.int64)
  (
    ahf_halo_id,
    ahf_parent_halo_id,
    ahf_top_halo_id,
    ahf_depth,
    caesar_parent_halo_index,
    caesar_top_halo_index,
  ) = _ahf_lineage_for_ids(tree, halo_ids, halo_ids)

  halo_data['AHF_haloID'] = ahf_halo_id
  halo_data['AHF_parent_haloID'] = ahf_parent_halo_id
  halo_data['AHF_top_haloID'] = ahf_top_halo_id
  halo_data['AHF_depth'] = ahf_depth
  halo_data['caesar_parent_halo_index'] = caesar_parent_halo_index
  halo_data['caesar_top_halo_index'] = caesar_top_halo_index
  halo_data['AHF_ancestor_haloIDs'] = pd.Series(
      _ahf_ancestor_chains(tree, halo_ids), index=halo_data.index, dtype=object
  )
  halo_data['child'] = caesar_parent_halo_index != -1

  galaxy_data = data_manager.group_data.get('galaxies')
  if galaxy_data is None or 'parent_halo_index' not in galaxy_data:
    return

  galaxy_source_halo_ids = galaxy_data['parent_halo_index'].to_numpy(dtype=np.int64)
  (
    ahf_halo_id,
    ahf_parent_halo_id,
    ahf_top_halo_id,
    ahf_depth,
    caesar_parent_halo_index,
    caesar_top_halo_index,
  ) = _ahf_lineage_for_ids(tree, galaxy_source_halo_ids, halo_ids)

  galaxy_data['AHF_haloID'] = ahf_halo_id
  galaxy_data['AHF_parent_haloID'] = ahf_parent_halo_id
  galaxy_data['AHF_top_haloID'] = ahf_top_halo_id
  galaxy_data['AHF_depth'] = ahf_depth
  galaxy_data['_ahf_host_halo_index'] = caesar_top_halo_index
  galaxy_data['caesar_parent_halo_index'] = caesar_parent_halo_index
  galaxy_data['caesar_top_halo_index'] = caesar_top_halo_index
  galaxy_data['AHF_ancestor_haloIDs'] = pd.Series(
      _ahf_ancestor_chains(tree, galaxy_source_halo_ids), index=galaxy_data.index, dtype=object
  )


def _assign_central_galaxies(data_manager: DataManager) -> None:
  halo_data = data_manager.group_data.get('halos')
  if halo_data is None:
    return
  if 'galaxies' not in data_manager.group_data:
    halo_data['central_galaxy'] = -1
    return

  galaxy_data = data_manager.group_data['galaxies']
  if len(galaxy_data) == 0 or 'parent_halo_index' not in galaxy_data or 'mass_star' not in galaxy_data:
    halo_data['central_galaxy'] = -1
    galaxy_data['central'] = False
    return

  parent_halo = galaxy_data['parent_halo_index'].to_numpy(dtype=np.int64)
  stellar_mass = galaxy_data['mass_star'].to_numpy(dtype=float)
  galaxy_ids = galaxy_data.index.to_numpy(dtype=np.int64)
  n_halos = len(halo_data)

  halo_positions = halo_data.index.get_indexer(parent_halo)
  central, central_by_halo = compute_central_galaxy_flags(
      halo_positions.astype(np.int64), stellar_mass, galaxy_ids, n_halos
  )

  halo_data['central_galaxy'] = central_by_halo
  galaxy_data['central'] = central


def common_group_properties(data_manager: DataManager, group_name: str, particle_type: str) -> None:
  """
  Computes properties common to all particle types.
  """
  config = data_manager.config
  groupID_key = config['groupIDs'][group_name]
  group_data = data_manager.group_data[group_name]
  n_groups = len(group_data)

  # -
  # step 1: extract arrays from datamanager so the entire operation can be vectorised
  # - 

  if particle_type == 'total':
      ptypes = config['ptypes']
  elif particle_type == 'baryon':
      ptypes = config['ptypes_baryon']
  else:
      ptypes = [particle_type]

  # ids: groups, halos, galaxies
  group_ids_list, halo_ids_list, gal_ids_list = [], [], []
  # physical quantities: mass, potential
  masses_list, potentials_list = [], []
  # kinematics: positions, velocities
  positions_list, velocities_list = [], []

  # cast the required data to numpy.
  for ptype in ptypes:
    df = data_manager.data[ptype]
    group_ids_list.append(df[groupID_key].to_numpy())
    halo_ids_list.append(df['HaloID'].to_numpy())
    gal_ids_list.append(df['GalID'].to_numpy())
    masses_list.append(df['mass'].to_numpy())
    potentials_list.append(df['potential'].to_numpy())
    positions_list.append(df[['x', 'y', 'z']].to_numpy())
    velocities_list.append(df[['vx', 'vy', 'vz']].to_numpy())

  # flat arrays
  # group_ids = np.concatenate([data_manager.data[ptype][groupID_key] for ptype in ptypes])
  # halo_ids = np.concatenate([data_manager.data[ptype]['HaloID'] for ptype in ptypes])
  # gal_ids = np.concatenate([data_manager.data[ptype]['GalID'] for ptype in ptypes])
  # masses = np.concatenate([data_manager.data[ptype]['mass'] for ptype in ptypes])
  # potentials = np.concatenate([data_manager.data[ptype]['potential'] for ptype in ptypes])
  # potentials = np.vstack([data_manager.data[ptype][['x', 'y', 'z']] for ptype in ptypes])
  # velocities = np.vstack([data_manager.data[ptype][['vx', 'vy', 'vz']] for ptype in ptypes])

  group_ids = np.concatenate(group_ids_list)
  halo_ids = np.concatenate(halo_ids_list)
  gal_ids = np.concatenate(gal_ids_list)
  masses = np.concatenate(masses_list)
  potentials = np.concatenate(potentials_list)
  # vstack is equivalent to concatenate here: but these are 3D so this makes the format more explicit
  positions = np.vstack(positions_list)
  velocities = np.vstack(velocities_list)

  # filter particles not assigned to a specific galaxy
  if group_name == 'galaxies':
    keep = gal_ids != -1
    group_ids = group_ids[keep]
    halo_ids = halo_ids[keep]
    masses = masses[keep]
    potentials = potentials[keep]
    positions = positions[keep]
    velocities = velocities[keep]

  # if a group is empty
  if len(masses) == 0:
    return

  # -
  # step 2: group indexing
  # -

  group_idx = group_data.index.get_indexer(group_ids) # build a list of particles where the value of each particle is its group ID

  # parent halo assignment
  if group_name == 'galaxies' and particle_type == 'total':
      parent = np.full(n_groups, -1, dtype=np.int64)
      for i in range(len(group_ids)):
          g = group_idx[i]
          if parent[g] == -1:
              parent[g] = halo_ids[i]
      group_data['parent_halo_index'] = parent

  # -
  # step 3: nparticles
  # - 

  counts = count_per_group(group_idx, n_groups) # number of particles per group
  group_data[f'n{particle_type}'] = counts

  # - 
  # step 4: masses
  # - 

  if particle_type == 'bh':
    group_mass = max_value_per_group(masses, group_idx, n_groups) # REVIEW: why?
    group_mass[~np.isfinite(group_mass)] = 0.
  else:
    group_mass = sum_per_group(masses, group_idx, n_groups)

  group_data[f'mass_{particle_type}'] = group_mass

  # -
  # step 5: minimum potential for halos
  # - 

  if group_name == 'halos' and particle_type == 'total':

    minimum_potential_idx = min_idx_per_group(potentials, group_idx, n_groups)
    valid = minimum_potential_idx >= 0 # mask all else away

    minimum_potential_position = np.full((n_groups, 3), np.nan)
    minimum_potential_velocity = np.full((n_groups, 3), np.nan)
    minimum_potential_position[valid] = positions[minimum_potential_idx[valid]]
    minimum_potential_velocity[valid] = velocities[minimum_potential_idx[valid]]

    for i, d in enumerate(['x', 'y', 'z']):
      group_data[f'minpot_{d}'] = minimum_potential_position[:, i]
      group_data[f'minpot_v{d}'] = minimum_potential_velocity[:, i]

  # - 
  # stage 6: centre-of-mass 
  # - 

  com_positions = np.zeros((n_groups, 3))
  com_velocities = np.zeros((n_groups, 3))

  for d in range(3):
    com_positions[:, d] = _safe_divide(sum_per_group(positions[:, d] * masses, group_idx, n_groups), group_mass)
    com_velocities[:, d] = _safe_divide(sum_per_group(velocities[:, d] * masses, group_idx, n_groups), group_mass)

  for i, d in enumerate(['x', 'y', 'z']):
    group_data[f'{d}_{particle_type}'] = com_positions[:, i]
  for i, d in enumerate(['x', 'y', 'z']):
    group_data[f'v{d}_{particle_type}'] = com_velocities[:, i]

  # -
  # stage 7: relative quantities
  # -

  # halos: potential well, galaxies: centre-of-mass
  if group_name == 'halos':
    if particle_type == 'total':
        ref_positions = minimum_potential_position
        ref_velocities = minimum_potential_velocity
    else:
        ref_positions = group_data[['minpot_x', 'minpot_y', 'minpot_z']].to_numpy()
        ref_velocities = group_data[['minpot_vx', 'minpot_vy', 'minpot_vz']].to_numpy()
  else:
      ref_positions = com_positions
      ref_velocities = com_velocities

  positions_rel = positions - broadcast_to_particles(ref_positions, group_idx)
  velocities_rel_com = velocities - broadcast_to_particles(com_velocities, group_idx)
  velocities_rel_ref = velocities - broadcast_to_particles(ref_velocities, group_idx)

  radii = np.linalg.norm(positions_rel, axis=1)

  # -
  # step 8: velocity dispersion
  # -

  disp_sums = sum_per_group(np.sum(velocities_rel_com**2, axis=1), group_idx, n_groups)
  vel_disps = np.sqrt(_safe_divide(disp_sums, counts))
  group_data[f'velocity_dispersion_{particle_type}'] = vel_disps

  # -
  # step 9: angular momentum and rotation
  # -

  L, ktot = compute_angular_momentum(positions_rel, velocities_rel_ref, masses, group_idx, n_groups)
  
  for i, d in enumerate(['x', 'y', 'z']):
    group_data[f'L{d}_{particle_type}'] = L[:, i]

  L_mag = np.linalg.norm(L, axis=1)
  group_data[f'L_{particle_type}'] = L_mag
  group_data[f'ALPHA_{particle_type}'] = np.arctan2(L[:, 1], L[:, 2])
  group_data[f'BETA_{particle_type}'] = np.arcsin(_safe_divide(L[:, 0], L_mag))

  counter_mass, krot, ktot = compute_rotation_quantities(
      positions_rel, velocities_rel_ref, masses, group_idx, L, n_groups
  )

  group_data[f'BoverT_{particle_type}'] = _safe_divide(2 * counter_mass, group_mass)
  group_data[f'kappa_rot_{particle_type}'] = _safe_divide(krot, ktot)

  angular_cols = [
    f'velocity_dispersion_{particle_type}',
    f'Lx_{particle_type}', f'Ly_{particle_type}', f'Lz_{particle_type}',
    f'BoverT_{particle_type}', f'kappa_rot_{particle_type}',
    ]
  
  # for small groups: set quantites = 0 as they are not meaningful (as done in original code)
  small = counts < 3
  for col in angular_cols:
      vals = group_data[col].to_numpy().copy()
      vals[small] = 0.
      group_data[col] = vals

  # -
  # step 10: radial quantities
  # -

  # from previous code
  quantiles = np.array([0.2, 0.5, 0.8])
  quantile_names = ['r20', 'half_mass', 'r80']

  radial_results, radius_rmax = compute_radial_quantiles_and_rmax(
      radii, masses, group_idx, n_groups, quantiles
  )

  for q, col_name in enumerate(quantile_names):
      group_data[f'radius_{particle_type}_{col_name}'] = radial_results[:, q]
  group_data[f'radius_{particle_type}_rmax'] = radius_rmax

  # - 
  # step 11: virial quantities for halos
  # - 

  if group_name == 'halos' and particle_type == 'total':
    # from previous code
    group_data['r200'] = data_manager.simulation['r200_factor'] * group_mass**(1/3)
    G_factor = unyt.G.to('(km**2 * kpc)/(Msun * s**2)').d
    group_data['circular_velocity'] = np.sqrt(G_factor * group_mass / group_data['r200'])
    group_data['temperature'] = 3.6e5 * (group_data['circular_velocity'] / 100.0)**2
    group_data['spin_param'] = _safe_divide(
        L_mag,
        np.sqrt(2) * group_mass * group_data['circular_velocity'].to_numpy() * group_data['r200'].to_numpy()
    )

    rhocrit = (
        data_manager.simulation['rhocrit'] *
        data_manager.create_unit_quantity('rhocrit')
    ).to('Msun / (kpc*a)**3').d

    factors = np.array([200., 500., 2500.])

    virial_r, virial_m = compute_virial_quantities(radii, masses, group_idx, n_groups, rhocrit, factors)

    for f, factor in enumerate([200, 500, 2500]):
        group_data[f'radius_{factor}_c'] = virial_r[:, f]
        group_data[f'mass_{factor}_c'] = virial_m[:, f]

def gas_group_properties(data_manager: DataManager, group_name: str) -> None:
    
  config = data_manager.config
  group_data = data_manager.group_data[group_name]
  groupID_key = config['groupIDs'][group_name]

  # load relevant quantities
  df = data_manager.data['gas']
  group_ids = df[groupID_key].to_numpy()
  masses = df['mass'].to_numpy()
  metallicities = df['metallicity'].to_numpy()
  sfrs = df['sfr'].to_numpy()
  temperatures = df['temperature'].to_numpy()
  rhos = df['rho'].to_numpy()
  mass_HI = df['mass_HI'].to_numpy()
  mass_H2 = df['mass_H2'].to_numpy()
  dust_masses = df['dustmass'].to_numpy() if 'dustmass' in df else np.zeros(len(df))

  if group_name == 'galaxies':
    keep = df['GalID'].to_numpy() != -1
    group_ids = group_ids[keep]
    masses = masses[keep]
    metallicities = metallicities[keep]
    sfrs = sfrs[keep]
    temperatures = temperatures[keep]
    rhos = rhos[keep]
    mass_HI = mass_HI[keep]
    mass_H2 = mass_H2[keep]
    dust_masses = dust_masses[keep]

  # guard against empty group
  if len(masses) == 0:
    return
  
  n_groups = len(group_data)
  group_idx = group_data.index.get_indexer(group_ids)
  order, unique_groups, start, end = sort_by_group(group_idx)
  (
    _group_mass_sum,
    gas_HI,
    gas_H2,
    dust_mass,
    ndust,
    sfr_sum,
    metal_mass,
    metal_sfr,
    temp_mass,
    cgm_mass_sum,
    cgm_temp_mass,
    cgm_temp_metal,
    cgm_metal_mass,
  ) = compute_gas_scalar_sums(
      unique_groups.astype(np.int64),
      start.astype(np.int64),
      end.astype(np.int64),
      masses[order],
      metallicities[order],
      sfrs[order],
      temperatures[order],
      rhos[order],
      mass_HI[order],
      mass_H2[order],
      dust_masses[order],
      n_groups,
      config['nHlim'],
  )

  # HI, H2 masses
  group_data['mass_HI'] = gas_HI
  group_data['mass_H2'] = gas_H2
  group_data['mass_dust'] = dust_mass
  group_data['ndust'] = ndust

  # CAESAR stores HI_ism/H2_ism from the same gas hydrogen sums in group_funcs.pyx.
  group_data['mass_HI_ism'] = gas_HI
  group_data['mass_H2_ism'] = gas_H2

  # SFR
  group_data['sfr'] = sfr_sum

  # metallicity
  group_mass = group_data[f'mass_gas'].to_numpy()
  group_sfr = group_data['sfr'].to_numpy()

  group_data['metallicity_mass_weighted'] = _safe_divide(metal_mass, group_mass)
  group_data['metallicity_sfr_weighted'] = _safe_divide(metal_sfr, group_sfr)

  # CGM quantities
  group_data['mass_cgm'] = cgm_mass_sum
  cgm_mass = group_data['mass_cgm'].to_numpy()

  # temperatures
  group_data['temp_mass_weighted'] = _safe_divide(temp_mass, group_mass)

  group_data['temp_mass_weighted_cgm'] = _safe_divide(cgm_temp_mass, cgm_mass)
  group_data['temp_metal_weighted_cgm'] = _safe_divide(cgm_temp_metal, cgm_temp_mass)

  # CGM metallicity
  group_data['metallicity_mass_weighted_cgm'] = _safe_divide(cgm_metal_mass, cgm_mass)
  group_data['metallicity_temp_weighted_cgm'] = _safe_divide(cgm_temp_metal, cgm_metal_mass)

def star_group_properties(data_manager: DataManager, group_name: str) -> None:
  config = data_manager.config
  group_data = data_manager.group_data[group_name]
  groupID_key = config['groupIDs'][group_name]

  df = data_manager.data['star']
  group_ids = df[groupID_key].to_numpy()
  masses = df['mass'].to_numpy()
  metallicities = df['metallicity'].to_numpy()
  ages = df['age'].to_numpy()

  if group_name == 'galaxies':
      keep = df['GalID'].to_numpy() != -1
      group_ids = group_ids[keep]
      masses = masses[keep]
      metallicities = metallicities[keep]
      ages = ages[keep]

  if len(masses) == 0:
      return

  n_groups = len(group_data)
  group_idx = group_data.index.get_indexer(group_ids)
  order, unique_groups, start, end = sort_by_group(group_idx)
  total_mass, metal_mass, age_mass, age_metal, young_mass = compute_star_scalar_sums(
      unique_groups.astype(np.int64),
      start.astype(np.int64),
      end.astype(np.int64),
      masses[order],
      metallicities[order],
      ages[order],
      n_groups,
  )

  # metallicity
  group_data['metallicity_stellar'] = _safe_divide(metal_mass, total_mass)

  # age
  group_mass_star = group_data['mass_star'].to_numpy()
  group_data['age_mass_weighted'] = _safe_divide(age_mass, group_mass_star)
  group_data['age_metal_weighted'] = _safe_divide(age_metal, metal_mass)
  group_data['sfr_100'] = young_mass / 100.e6

def bh_group_properties(data_manager: DataManager, group_name: str) -> None:
  config = data_manager.config
  group_data = data_manager.group_data[group_name]
  groupID_key = config['groupIDs'][group_name]

  df = data_manager.data['bh']
  group_ids = df[groupID_key].to_numpy()
  masses = df['mass'].to_numpy()
  bhmdots = df['bhmdot'].to_numpy()

  if group_name == 'galaxies':
      keep = df['GalID'].to_numpy() != -1
      group_ids = group_ids[keep]
      masses = masses[keep]
      bhmdots = bhmdots[keep]

  if len(masses) == 0:
      return

  n_groups = len(group_data)
  group_idx = group_data.index.get_indexer(group_ids)

  # find most massive BH per group
  max_idx = max_idx_per_group(masses, group_idx, n_groups)
  valid = max_idx >= 0

  # bhmdot of most massive BH
  bhmdot = np.full(n_groups, np.nan)
  bhmdot[valid] = bhmdots[max_idx[valid]]
  group_data['bhmdot'] = bhmdot

  # Eddington fraction
  FRAD = 0.1
  edd_factor = (4 * np.pi * const.G * const.m_p / (FRAD * const.c * const.sigma_T)).to('1/yr').value

  bh_mass = np.full(n_groups, np.nan)
  bh_mass[valid] = masses[max_idx[valid]]
  group_data['bh_fedd'] = _safe_divide(bhmdot, edd_factor * bh_mass)

def calculate_local_densities(data_manager: DataManager) -> None:

  config = data_manager.config
  groups = config['groups']

  for group in groups:
    group_data = data_manager.group_data[group]

    if len(group_data) == 0:
      print(f"No group data!")
      continue

    pos = group_data[['x_total', 'y_total', 'z_total']].to_numpy()
    mass = group_data['mass_total'].to_numpy()

    # REVIEW: moving a FOF6D optimisation into this function
    # previously the pandas .explode() calls were memory-intensive
    boxsize = data_manager.simulation['boxsize']
    pos = np.where(pos > boxsize, pos - boxsize, pos)
    pos = np.where(pos < 0, pos + boxsize, pos)
    tree = KDTree(pos, boxsize=boxsize)
    for radius in [300., 1000., 3000.]:
      volume = 4./3. * np.pi * radius**3
      index_lists = tree.query_ball_point(pos, radius, workers=-1) # workers=-1 means all processors are used (from documentation)
      mass_sums = np.array([mass[il].sum() for il in index_lists])
      counts = np.array([len(il) for il in index_lists])

      group_data[f'local_mass_density_{int(radius)}'] = mass_sums / volume
      group_data[f'local_number_density_{int(radius)}'] = counts / volume


def _flatten_neighbor_lists(neighbor_lists):
    offsets = np.zeros(len(neighbor_lists) + 1, dtype=np.int64)
    for i, neighbours in enumerate(neighbor_lists):
        offsets[i + 1] = offsets[i] + len(neighbours)

    indices = np.empty(offsets[-1], dtype=np.int64)
    for i, neighbours in enumerate(neighbor_lists):
        start = offsets[i]
        end = offsets[i + 1]
        if end > start:
            indices[start:end] = neighbours

    return offsets, indices


def calculate_aperture_masses(data_manager, config):
    
    group_data = data_manager.group_data['galaxies']
    n_galaxies = len(group_data)
    galaxy_pos = group_data[['x_total', 'y_total', 'z_total']].to_numpy()
    parent_halo = group_data['parent_halo_index'].to_numpy()
    aperture = 30. # as defined previously

    # use helper function
    include_hydrogen = 'gas' in config['ptypes'] and {'mass_HI', 'mass_H2'}.issubset(data_manager.data['gas'].columns)
    include_dust = 'gas' in config['ptypes'] and 'dustmass' in data_manager.data['gas'].columns
    all_pos, all_mass, all_codes, all_halos, ptype_names, all_vel = extract_particle_arrays(
        data_manager, config, include_hydrogen=include_hydrogen, include_dust=include_dust, include_velocities=True
    )
    # may want to check the helper function include_hydrogen part, thought it might be useful in future
    n_ptypes = len(ptype_names)
    output_names = ptype_names + ['total', 'baryon']
    output_index = {name: i for i, name in enumerate(output_names)}
    include_matrix = np.zeros((len(output_names), n_ptypes), dtype=np.bool_)
    for i in range(n_ptypes):
        include_matrix[i, i] = True
    for i, name in enumerate(ptype_names):
        if name in config['ptypes']:
            include_matrix[output_index['total'], i] = True
        if name in config['ptypes_baryon']:
            include_matrix[output_index['baryon'], i] = True

    # pre-sort particles by halo
    order, unique_halos, h_start, h_end = sort_by_group(all_halos)
    all_pos = all_pos[order]
    all_mass = all_mass[order]
    all_codes = all_codes[order]
    all_vel = all_vel[order]

    # pre-sort galaxies by parent halo
    gal_order, halos_with_galaxies, gal_start, gal_end = sort_by_group(parent_halo)

    result = np.zeros((n_galaxies, len(output_names)))
    velocity_result = np.zeros((n_galaxies, len(output_names)))
    boxsize = data_manager.simulation['boxsize']

    for h in range(len(unique_halos)):
        halo_id = unique_halos[h]
        halo_pos = all_pos[h_start[h]:h_end[h]]
        halo_mass = all_mass[h_start[h]:h_end[h]]
        halo_codes = all_codes[h_start[h]:h_end[h]]
        halo_vel = all_vel[h_start[h]:h_end[h]]

        # guard
        if len(halo_pos) == 0:
            continue

        gh_idx = np.searchsorted(halos_with_galaxies, halo_id) # find where the hid sits in halos_with_galaxies
        if gh_idx >= len(halos_with_galaxies) or halos_with_galaxies[gh_idx] != halo_id:
            continue # skip halos with no galaxies
        gal_indices = gal_order[gal_start[gh_idx]:gal_end[gh_idx]]

        # guard
        if len(gal_indices) == 0:
            continue

        # build KDTree (explained in FOF6D code)
        halo_pos_wrapped = np.mod(halo_pos, boxsize)
        galaxy_pos_wrapped = np.mod(galaxy_pos[gal_indices], boxsize)
        tree = KDTree(halo_pos_wrapped, boxsize=boxsize)
        try:
            neighbor_lists = tree.query_ball_point(galaxy_pos_wrapped, aperture, workers=-1)
        except TypeError:
            neighbor_lists = tree.query_ball_point(galaxy_pos_wrapped, aperture)

        neighbor_offsets, neighbor_indices = _flatten_neighbor_lists(neighbor_lists)
        masses_local, sigmas_local = compute_aperture_component_properties(
            neighbor_offsets,
            neighbor_indices,
            halo_mass,
            halo_codes.astype(np.int64),
            halo_vel,
            include_matrix,
        )
        result[gal_indices, :] = masses_local
        velocity_result[gal_indices, :] = sigmas_local

    for i, name in enumerate(ptype_names):
        group_data[f'mass_{name}_30kpc'] = result[:, i]
        group_data[f'velocity_dispersion_{name}_30kpc'] = velocity_result[:, i]

    group_data['mass_total_30kpc'] = result[:, output_index['total']]
    group_data['velocity_dispersion_total_30kpc'] = velocity_result[:, output_index['total']]
    group_data['velocity_dispersion_baryon_30kpc'] = velocity_result[:, output_index['baryon']]

def calculate_group_properties(data_manager: DataManager) -> None:
  data_manager.logger.info('Calculating group properties...')
  t1 = perf_counter()

  # admin
  config = data_manager.config

  t2 = perf_counter()
  for ptype in config['ptypes']:
    data_manager.load_property('potential', ptype)
  t3 = perf_counter()
  t4 = t5 = t6 = t7 = t8 = t9 = t3

  groups = config['groups']

  to_process = config['to_process']
  aperture_velocities_needed = 'apertures' in to_process and 'galaxies' in config['groups']

  # unnecessary columns
  columns_to_drop = ['potential'] if aperture_velocities_needed else ['vx', 'vy', 'vz', 'potential']

  # order to iterate over
  ptype_order = ['total', 'dm', 'baryon', 'gas', 'star', 'bh']
  # drop the necessary columns afterwards
  drop_after = {'dm', 'gas', 'star', 'bh'}

  # common group properties
  for ptype in ptype_order:
      if ptype not in to_process:
          continue
      for group in groups:
          common_group_properties(data_manager, group, ptype)
      if ptype in drop_after:
          data_manager.data[ptype].drop(columns=columns_to_drop, inplace=True, errors='ignore')

  _assign_central_galaxies(data_manager)

  # gas properties
  if 'gas' in to_process:
    t4 = perf_counter()
    for property in ['rho', 'nh', 'fH2', 'metallicity', 'sfr', 'temperature']:
      data_manager.load_property(property, 'gas')
    data_manager.load_property('dustmass', 'gas', optional=True, default=0.)
    t5 = perf_counter()

    # gas masses
    data = data_manager.data['gas']
    data['fHI'] = _safe_divide(data['nh'].to_numpy(), data['mass'].to_numpy(), fill=0.)
    not_conserving_mass = data.eval('(fHI + fH2) > 1')
    data.loc[not_conserving_mass, 'fHI'] = 1. - data.loc[not_conserving_mass, 'fH2']

    data['mass_HI'] = config['XH'] * data['fHI'] * data['mass']
    data['mass_H2'] = config['XH'] * data['fH2'] * data['mass']

    for group in groups:
      gas_group_properties(data_manager, group)

  # star properties
  if 'star' in to_process:
    t6 = perf_counter()
    for property in ['age', 'metallicity']:
      data_manager.load_property(property, 'star')
    t7 = perf_counter()

    for group in groups:
      star_group_properties(data_manager, group)

  # bh properties
  if 'bh' in to_process:
    t8 = perf_counter()
    for property in ['bhmdot']:
      data_manager.load_property(property, 'bh')
    t9 = perf_counter()
      
    for group in groups:
      bh_group_properties(data_manager, group)

  # apertures
  if 'apertures' in to_process and 'galaxies' in config['groups']:
    calculate_aperture_masses(data_manager, config)

  if aperture_velocities_needed:
    for ptype in drop_after:
      if ptype in data_manager.data:
        data_manager.data[ptype].drop(columns=['vx', 'vy', 'vz'], inplace=True, errors='ignore')

  # densities
  if 'local_densities' in to_process:
    calculate_local_densities(data_manager)

  _assign_halo_source_properties(data_manager)

  t10 = perf_counter()
  data_manager.logger.info(f'Calculating group properties done in {t10-t1:.2f} seconds. (Load reduced: {t10-t9 + t8-t7 + t6-t5 + t4-t3 + t2-t1:.2f} seconds.)')
