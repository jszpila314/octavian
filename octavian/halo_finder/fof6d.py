from __future__ import annotations
from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
  from octavian.data_manager import DataManager

import numpy as np
import pandas as pd
import unyt
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed

from time import perf_counter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# get mis for fof6d
def get_mean_interparticle_separation(data_manager: 'DataManager') -> None:
  t = data_manager.simulation['time']
  a = data_manager.simulation['a']
  h = data_manager.simulation['h']
  Om = data_manager.simulation['O0']
  boxsize = data_manager.simulation['boxsize']

  GRAV = unyt.G.to('cm**3/(g*s**2)').d
  UL = (1. * unyt.kpc).to('cm').d
  UM = data_manager.create_unit_quantity('mass').to('g').d
  UT = t/a

  G = GRAV / UL**3 * UM * UT**2
  Hubble = 3.2407789e-18 * UT

  dmmass = data_manager.mdm_total
  ndm = data_manager.ndm

  gmass = data_manager.mgas_total
  smass = data_manager.mstar_total
  bhmass = data_manager.mbh_total

  bmass = gmass + smass + bhmass

  Ob = bmass / (bmass + dmmass) * Om
  rhodm = (Om - Ob) * 3.0 * Hubble**2 / (8.0 * np.pi * G) / h

  mis = ((dmmass / ndm / rhodm)**(1./3.))/h
  efres = int(boxsize/h/mis)

  data_manager.mis = mis
  data_manager.efres = efres
  data_manager.Ob = Ob


# initial assignment of galaxy ids through sorting in x,y,z directions
def fof_sort_halo(halo: 'pd.DataFrame', minstars: int, fof_LL: float) -> 'pd.DataFrame':
  for direction in ['x', 'y', 'z']:
    halo = halo.sort_values(by=['GalID', direction])
    halo['distance'] = np.diff(halo[direction], prepend=halo[direction].iloc[0])
    halo['GalID'] += np.cumsum(halo['distance'] > fof_LL)

  halo = halo.groupby(by='GalID').filter(lambda group: len(group) >= minstars)

  return halo

#
# helper functions
#

# kernel table for fof6d velocity criterion distance weights
def create_kernel_table(fof_LL,ntab=1000):
    kerneltab = np.zeros(ntab+1)
    hinv = 1./fof_LL
    for i in range(ntab):
        r = 1. * i / ntab
        q = 2 * r * hinv
        if q > 2: kerneltab[i] = 0.0
        elif q > 1: kerneltab[i] = 0.25 * (2 - q)**3
        else: kerneltab[i] = 1 - 1.5 * q * q * (1 - 0.5 * q)
    return kerneltab

# kernel table lookup
def kernel(r_over_h,kerneltab):
    ntab = len(kerneltab) - 1
    rtab = ntab * r_over_h + 0.5
    itab = rtab.astype(int)
    return kerneltab[itab]

#
# fof6d functions
#

# fof6d function to apply on groups
def run_fof6d_in_halo(halo: pd.DataFrame, kernel_table: np.ndarray, minstars: int, fof_LL: float, vel_LL: Optional[float] = None) -> list[list[tuple[int, pd.Index]]]:

  timings = {'n_particles': len(halo)}
  if len(halo) < minstars:
    return [], timings
  
  t0 = perf_counter()

  # stage 1: directional group find
  halo = fof_sort_halo(halo, minstars, fof_LL)
  timings['sort'] = perf_counter() - t0
  groups = [halo.loc[halo['GalID'] == id] for id in halo['GalID'].unique()]

  if len(groups) == 0:
    return [], timings

  # skip stage 2 if vel_LL not defined, all members of a group form a galaxy
  if vel_LL is None:
    galaxies = [[(i, group_ptype.index) for i, group_ptype in group.groupby(by='ptype')] for group in groups]
    return galaxies, timings
  

  #
  # stage 2: fof6d
  #
  # updated vectorised loop implementation (JP)
  #

  # diagnostics
  t0 = perf_counter() # reset the clock
  t_neighbors_tot = 0
  t_weights_tot = 0
  t_merge_tot = 0

  new_groups = []
  for group in groups:
    t1 = perf_counter()
    # NOTE: original implementation of nearestneighbours
    pos = group[['x', 'y', 'z']].to_numpy() 
    neighbors = NearestNeighbors(radius=fof_LL)
    neighbors.fit(pos)
    neighborDistances_lists, index_lists = neighbors.radius_neighbors(pos)
    t2 = perf_counter()
    t_neighbors_tot += t2 - t1 # profiling: nearest neighbour speeds
    vel = group[['vx', 'vy', 'vz']].to_numpy()

    # NOTE: new algorithm using a sparse matrix implementation to avoid expensive python loops and pass code into scipy's C
    # vectorised COO (COOrdinate) construction (scipy recommended)
    n = len(group)
    lengths = np.array([len(il) for il in index_lists])
    # meet the definition of a sparse matrix
    # row[i], col[i] = value[i]
    rows = np.repeat(np.arange(n), lengths) 
    cols = np.concatenate(index_lists)
    dists = np.concatenate(neighborDistances_lists)

    # vectorised kernel weights (adapted from Jakub)
    q = dists / fof_LL
    w = kernel(q, kernel_table)  # already works on arrays

    # vectorised velocity differences
    vel_diff = np.linalg.norm(vel[cols] - vel[rows], axis=1)

    # vectorised sigma per particle
    weighted_dv_sq = w * vel_diff**2 # same as Jakub (I renamed variables for readability)
    sigmas = np.sqrt(np.bincount(rows, weights=weighted_dv_sq, minlength=n)) 
    t3 = perf_counter()
    t_weights_tot += t3 - t2 # profiling: weighting 

    # vectorised velocity criterion
    valid = vel_diff <= (vel_LL * sigmas[rows])

    # build sparse matrix from valid edges only
    # REVIEW: csr matrices
    # https://stackoverflow.com/questions/11016256/connected-components-in-a-graph-with-100-million-nodes (the syntax has changed slightly with new scipy versions)
    adj = csr_matrix((np.ones(valid.sum()), (rows[valid], cols[valid])), shape=(n, n)) # np.ones matrix; boolean mask with rows, cols
    n_components, labels = connected_components(adj, directed=False) # directed=False means we only care about connections (preserves original logic)
    del adj # FIXME: not necessary?

    t4 = perf_counter()
    t_merge_tot += t4 - t3 # profiling: merging (big python loop no more)

    # unavoidable python loop
    for label in range(n_components):
        indices = np.where(labels == label)[0]
        if len(indices) < minstars:
            continue
        galaxy = group.iloc[indices]
        if len(galaxy.loc[galaxy['ptype'] == 'star']) >= minstars:
            new_groups.append(galaxy)

  # profiling: timings
  timings['neighbors'] = t_neighbors_tot
  timings['weights'] = t_weights_tot
  timings['merge'] = t_merge_tot

  # unavoidable python loop
  galaxies = [[(i, group_ptype.index) for i, group_ptype in group.groupby(by='ptype')] for group in new_groups]

  return galaxies, timings

# vectorised version of caesar fof6d
def run_fof6d(data_manager: DataManager, nproc: int = 1) -> None:
  config = data_manager.config

  for ptype in config['ptypes']:
    data_manager.load_property('mass', ptype)

  data_manager.mdm_total = np.sum(data_manager.data['dm']['mass'])
  data_manager.ndm = len(data_manager.data['dm'])

  data_manager.mgas_total = 0. if 'gas' not in config['ptypes'] else np.sum(data_manager.data['gas']['mass'])
  data_manager.mstar_total = 0. if 'star' not in config['ptypes'] else np.sum(data_manager.data['star']['mass'])
  data_manager.mbh_total = 0. if 'bh' not in config['ptypes'] else np.sum(data_manager.data['bh']['mass'])

  get_mean_interparticle_separation(data_manager)

  b = 0.02
  fof_LL = data_manager.mis * b
  vel_LL = 1.

  for ptype in config['ptypes']:
    data_manager.load_property('vel', ptype)

  # check dense
  for prop in ['rho', 'temperature', 'sfr']:
    data_manager.load_property(prop, 'gas')

  data_manager.data['gas']['temperature'] = 0.
  data_manager.data['gas']['dense_gas'] = (data_manager.data['gas']['rho'] > config['nHlim']) & ((data_manager.data['gas']['temperature'] < config['Tlim']) | (data_manager.data['gas']['sfr'] > 0))
  
  # combine dfs, reduce the gas df to common columns
  fof_columns = ['HaloID', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'ptype']
  fof_filter = lambda halo: len(halo) >= config['MINIMUM_STARS_PER_GALAXY']
  fof_halos = data_manager.data['star'].groupby('HaloID', observed=True).filter(fof_filter)
  fof_haloids = np.unique(fof_halos['HaloID'])
  fof_halos = pd.concat([data_manager.data['gas'].loc[data_manager.data['gas']['dense_gas'], fof_columns], data_manager.data['star'][fof_columns], data_manager.data['bh'][fof_columns]]).query('HaloID in @fof_haloids')

  fof_halos['GalID'] = 0
  kernel_table = create_kernel_table(fof_LL)
  grouped = fof_halos.groupby(by='HaloID', observed=True)

  results = Parallel(n_jobs=nproc)(delayed(run_fof6d_in_halo)(halo, kernel_table, config['MINIMUM_STARS_PER_GALAXY'], fof_LL, vel_LL) for idx, halo in grouped)
  galaxies = [g for gals, _ in results for g in gals if len(gals) != 0]
  all_timings = [t for _, t in results]
  timings_df = pd.DataFrame(all_timings)
  
  for ptype in config['ptypes']:
    data_manager.data[ptype]['GalID'] = -1
  
  for i, galaxy in enumerate(galaxies):
    for ptype, ptype_indexes in galaxy:
      data_manager.data[ptype].loc[ptype_indexes, 'GalID'] = i

  for ptype in config['ptypes']:
    data_manager.data[ptype]['GalID'] = data_manager.data[ptype]['GalID'].astype('category')

  print(f"\n=== FOF6D Timing Summary (rank {data_manager.rank}) ===")
  print(f"Halos processed: {len(timings_df)}")
  print(f"Total particles: {timings_df['n_particles'].sum()}")
  print(timings_df[['sort', 'neighbors', 'weights', 'merge']].sum().to_string())
  print(f"\nTop 5 halos by total time:")
  timings_df['total'] = timings_df[['sort', 'neighbors', 'weights', 'merge']].sum(axis=1)
  print(timings_df.nlargest(5, 'total').to_string())
  print("=" * 40)