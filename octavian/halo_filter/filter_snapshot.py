import h5py
import numpy as np


def get_id_filter(f: h5py.File, ptypes: list[str], nsplit: int) -> list[list[int]]:
  ids = []
  for ptype in ptypes:
    ids_ptype = f[ptype]['HaloID'][:]
    ids.append(ids_ptype[ids_ptype != 0])

  ids = np.sort(np.concatenate(ids))
  unique_ids, counts = np.unique(ids, return_counts=True)
  cumulative_counts = np.cumsum(counts)
  total = len(ids)

  split_ids = [0]
  split_fractions = np.linspace(0., 1., nsplit + 1)
  split_fractions = split_fractions[1:]

  for fraction in split_fractions:
    fraction_count = total * fraction
    split_ids.append(unique_ids[(np.abs(cumulative_counts - fraction_count)).argmin()])

  id_filter = list(zip(split_ids[:-1], split_ids[1:]))

  return id_filter


def filter_snapshot(snapfile: str, outfile: str, nsplit: int=4):
  with h5py.File(snapfile, 'r') as f:
    for i in range(nsplit):
      with h5py.File(f'{outfile}_{i}.hdf5', 'a') as f_out:
        f.copy(f['Header'], f_out, 'Header')


    ptypes = [group for group in list(f.keys()) if 'HaloID' in list(f[group].keys())]
    id_filter = get_id_filter(f, ptypes, nsplit)

    for ptype in ptypes:
      datasets = list(f[ptype].keys())

      ids = f[ptype]['HaloID'][:]
      particle_index = np.arange(len(ids), dtype='int')

      in_halo = ids != 0
      ids = ids[in_halo]

      order = np.argsort(ids)
      ids = ids[order]

      datasets = datasets + ['particle_index']

      for dataset in datasets:
        print(ptype, dataset)
        if dataset == 'particle_index':
          data = particle_index[in_halo][order]
        else:
          data = f[ptype][dataset][:][in_halo][order]

        for i, (start, end) in enumerate(id_filter):
          with h5py.File(f'{outfile}_{i}.hdf5', 'a') as f_out:
            f_out.require_group(ptype)
            in_halos = np.logical_and(ids > start, ids <= end)
            f_out[ptype][dataset] = data[in_halos]

