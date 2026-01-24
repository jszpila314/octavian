import caesar
print(f"caesar imported successfully.")
import pygadgetreader
print(f"pygadgetreader imported successfully.")
import fsps
print(f"fsps imported successfully.")
import octavian
print(f"octavian imported successfully.")
print(f"Packages imported successfully!")


path_to_snapshot = "/home/jp_duminy/SHP_Code/snap_m25n512_062.hdf5"
path_to_output = "/home/jp_duminy/SHP_Code/snapshot_filtered.hdf5"
path_to_config = "/home/jp_duminy/SHP_Code/octavian-1/config.yaml"

print(f"Attempting to filter snapshot...")

octavian.filter_snapshot(path_to_snapshot, path_to_output)

print(f"Success.")

print(f"Running Octavian...")
                        
octavian.run(path_to_snapshot, path_to_output, path_to_config)

print(f"Success.")
            