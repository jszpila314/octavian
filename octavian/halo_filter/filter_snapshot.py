import h5py
import numpy as np

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def filter_snapshot(snapfile: str, outfile: str):
  with h5py.File(snapfile, 'r') as f:
    for i in range(4):
      with h5py.File(f'{outfile}_{i}.hdf5', 'a') as f_out:
        f.copy(f['Header'], f_out, 'Header')


    ptypes = [group for group in list(f.keys()) if 'HaloID' in list(f[group].keys())]
    ids = []
    for ptype in ptypes:
      ids_ptype = f[ptype]['HaloID'][:]
      ids.append(ids_ptype[ids_ptype != 0])

    ids = np.sort(np.concat(ids))
    unique_ids, counts = np.unique(ids, return_counts=True)
    cumulative_counts = np.cumsum(counts)
    total = len(ids)

    split_ids = [0]
    for fraction in [0.25, 0.5, 0.75, 1]:
      fraction_count = total * fraction
      split_ids.append(unique_ids[(np.abs(cumulative_counts - fraction_count)).argmin()])

    id_filter = list(zip(split_ids[:-1], split_ids[1:]))

    for ptype in ptypes:
      datasets = list(f[ptype].keys())
      ids = f[ptype]['HaloID'][:]
      in_halo = ids != 0
      ids = ids[in_halo]

      order = np.argsort(ids)
      ids = ids[order]

      for dataset in datasets:
        print(ptype, dataset)
        data = f[ptype][dataset][:][in_halo][order]

        for i, (start, end) in enumerate(id_filter):
          with h5py.File(f'{outfile}_{i}.hdf5', 'a') as f_out:
            f_out.require_group(ptype)
            in_halos = np.logical_and(ids > start, ids <= end)
            f_out[ptype][dataset] = data[in_halos]

