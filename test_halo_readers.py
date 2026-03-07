# test_ahf_reader.py
from octavian.data_manager import DataManager
print(f"Data manager imported.")
from octavian.halo_reader.ahf import load_ahf
print(f"AHF loader imported.")
from yaml import safe_load
print(f"YAML imported.")
from time import perf_counter

print(f"Beginning analysis.")

with open('config.yaml', 'r') as f:
    config = safe_load(f)
config['Tlim'] = float(config['Tlim'])
config['prop_aliases']['pid'] = 'ParticleIDs'

snapshot = '/disk04/rad/sim/m100n1024/s50/snap_m100n1024_151.hdf5'
dm = DataManager(snapshot, config)

particles_path = '/home/jpduminy/octavian-snapshots/Simba_M200_snap_151.z0.000.AHF_particles'
halos_path = '/home/jpduminy/octavian-snapshots/Simba_M200_snap_151.z0.000.AHF_halos'

t1 = perf_counter()
print('Loading AHF...')
load_ahf(dm, particles_path, halos_path=halos_path, mode='field')
t2 = perf_counter()
time_taken = (t2 - t1) / 60 # mins
print(f"AHF Data loaded. Total time: {time_taken:.3f} minutes.")


for ptype in dm.config['ptypes']:
    assigned = (dm.data[ptype]['HaloID'] != -1).sum()
    total = len(dm.data[ptype])
    fraction = assigned / total
    print(f"{ptype}:")
    print(  f"{assigned}/{total} assigned ({fraction} %)")

print(f'\nField halos: {len(dm.halo_tree._field_halos)}')
print(f'Total halos: {len(dm.halo_tree.halo_ids)}')
print(f'Max depth: {dm.halo_tree.depths.max()}')