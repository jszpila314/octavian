"""

Here you will find the halo data management classes.

HaloReader ties the whole thing together. It is an agnostic class which handles all the data management
from a specified halo finder. The goal of the halo-finder-specific files is to then translate their outputs
into the format this class expects.

HaloTree provides the framework for storing substructure (subhalos, sub-subhalos, etc.) that come from
halo finders such as AHF and HBT+. It is useful to do this thoroughly for things like progenitors, and
because it allows us to capture more science data rather than tossing everything.

HaloMembership characterises the existing halos.

I think OOP is good here because bespoke halo readers can use inheritance.

"""

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from octavian.data_manager import DataManager

import numpy as np
import pandas as pd
from numba import njit


# the readers extract Octavian-compatible particles
# {'gas': 0, 'dm': 1, 'star': 2, 'bh': 3} is our convention for what readers must return
# readers must also align halo IDs so they read 0, 1, 2 etc.

# easier to work with integers than strings
PTYPE_ENCODE = {'gas': 0, 'dm': 1, 'star': 2, 'bh': 3}
PTYPE_DECODE = {i: j for j, i in PTYPE_ENCODE.items()} # the inverse operation

class HaloReader:
    """
    Base class for external halo finder integration.
    
    Subclasses use read() to parse their specific file formats.

    This class handles remapping, tree construction, snapshot matching, 
    and DataManager assignment for Octavian-friendly analysis.
    """

    def __init__(self, data_manager: DataManager):
        self.dm = data_manager
        self.tree = None
        self.membership = None

    def remap_ids(self, halo_ids, parent_ids, member_hids):
        """
        Map halo-finder-specific IDs to an agnostic 0, 1, 2 (speeds up agnostic classes, easier to work with).
        """
        unique_raw = np.unique(halo_ids)
        raw_to_new = np.full(unique_raw.max() + 1, -1, dtype=np.int64)
        raw_to_new[unique_raw] = np.arange(len(unique_raw))

        new_halo_ids = raw_to_new[halo_ids]

        # parents: map known IDs, anything else is -1 as is Octavian tradition
        new_parent_ids = np.full_like(parent_ids, -1)
        valid_parents = parent_ids != -1 
        parent_in_range = valid_parents & (parent_ids <= unique_raw.max())
        mapped = raw_to_new[parent_ids[parent_in_range]]
        # only assign if the parent actually exists in our halo list
        new_parent_ids[parent_in_range] = np.where(mapped != -1, mapped, -1)

        # membership particle halo IDs
        new_member_hids = raw_to_new[member_hids]

        return new_halo_ids, new_parent_ids, new_member_hids

    def assign(self, membership, mode):
        """
        Resolve memberships, match to snapshot, assign HaloIDs.
        """
        pids, ptypes, hids = membership.branch_membership(mode=mode)

        config = self.dm.config

        for ptype in config['ptypes']:
            ptype_code = PTYPE_ENCODE.get(ptype)
            if ptype_code is None:
                continue

            # filter to this ptype
            mask = ptypes == ptype_code
            if not np.any(mask):
                self.dm.data[ptype]['HaloID'] = pd.Series(
                    np.full(len(self.dm.data[ptype]), -1, dtype=np.int64), 
                    dtype='category'
                )
                continue

            ext_pids = pids[mask]
            ext_hids = hids[mask]

            self.match_ptype(ptype, ext_pids, ext_hids)

    def match_ptype(self, ptype, ext_pids, ext_hids):
        """
        Searchsorted matching of external particle IDs against snapshot.
        """
        self.dm.load_property('pid', ptype)
        snap_pids = self.dm.data[ptype]['pid'].to_numpy(dtype=np.int64)

        # sort external for searchsorted
        order = np.argsort(ext_pids)
        sorted_pids = ext_pids[order]
        sorted_hids = ext_hids[order]

        # match
        positions = np.searchsorted(sorted_pids, snap_pids)
        positions = np.clip(positions, 0, len(sorted_pids) - 1)
        matched = sorted_pids[positions] == snap_pids

        # assign — unmatched particles get -1
        halo_ids = np.full(len(snap_pids), -1, dtype=np.int64)
        halo_ids[matched] = sorted_hids[positions[matched]]

        self.dm.data[ptype]['HaloID'] = pd.Series(halo_ids, dtype='category')

class HaloTree:
    """
    Handles halo hierarchy for a single snapshot.
    """

    def __init__(self, halo_ids, parent_ids, properties=None):

        # base structure
        self.halo_ids = np.asarray(halo_ids, dtype=np.int64)
        self.parent_ids = np.asarray(parent_ids, dtype=np.int64)
        self.properties = properties

        # guard against no halos being present
        if len(self.halo_ids) == 0:
            print(f"No halos found.")
            self._id_to_idx = np.empty(0, dtype=np.int32)
            self.depths = np.empty(0, dtype=np.int32)
            self.field_map = np.empty(0, dtype=np.int64)
            self._depth_lookup = np.empty(0, dtype=np.int32)
            return

        # hid to index: map halo ID to its place in the hierarchy
        max_id = self.halo_ids.max()
        self._id_to_idx = np.full(max_id + 1, -1, dtype=np.int32) # array of -1s with len(nhalos)
        self._id_to_idx[self.halo_ids] = np.arange(len(self.halo_ids)) # val[i] = unique halo

        # halo structuring
        # simply sorts halos and their subhalos by hid
        order = np.argsort(self.parent_ids)
        self._parents_sorted = self.parent_ids[order]
        self._children_sorted = self.halo_ids[order]

        # csr format
        changes = np.flatnonzero(np.diff(self._parents_sorted)) + 1 # +1 helps because np.diff technically shifts the array
        self._child_offsets = np.concatenate(([0], changes))
        self._child_lengths = np.diff(np.concatenate((self._child_offsets, [len(self._parents_sorted)])))
        self._child_parent_keys = self._parents_sorted[self._child_offsets]

        self._parent_to_child_idx = np.full(max_id + 1, -1, dtype=np.int32)
        self._parent_to_child_idx[self._child_parent_keys] = np.arange(len(self._child_parent_keys))
        
        # compute on instantiation
        # both passed to numba functions
        self.depths = compute_depths(self.halo_ids, self.parent_ids, self._id_to_idx)
        self.field_map = build_field_map(self.halo_ids, self.parent_ids, self._id_to_idx)

        self._depth_lookup = np.zeros(max_id + 1, dtype=np.int32)
        self._depth_lookup[self.halo_ids] = self.depths

        # field halos
        field_mask = self.parent_ids == -1
        self._field_halos = self.halo_ids[field_mask]

    def get_depth(self, halo_id):
        """
        Returns the depth of a halo (0 = field halo)
        """
        return self._depth_lookup[halo_id]

    def get_children(self, halo_id):
        """
        Returns children of a halo.
        """
        idx = self._parent_to_child_idx[halo_id]
        if idx == -1:
            return np.empty(0, dtype=np.int64)
        start = self._child_offsets[idx]
        end = start + self._child_lengths[idx]
        return self._children_sorted[start:end]

class HaloMembership:
    """
    Handles halo membership for a single snapshot.
    """

    def __init__(self, tree: HaloTree, halo_ids, particle_ids, ptype_codes):

        self.tree = tree

        # guard for no halos
        if len(halo_ids) == 0:
            print(f"No halos found.")
            self._member_hids = np.empty(0, dtype=np.int64)
            self._member_pids = np.empty(0, dtype=np.int64)
            self._member_ptypes = np.empty(0, dtype=np.int8)
            self._offsets = np.empty(0, dtype=np.int64)
            self._lengths = np.empty(0, dtype=np.int32)
            self._unique_hids = np.empty(0, dtype=np.int64)
            self._hid_to_idx = np.empty(0, dtype=np.int32)
            return

        # sort by hID for csr structure
        order = np.argsort(halo_ids)
        self._member_hids = halo_ids[order]
        self._member_pids = particle_ids[order]
        self._member_ptypes = ptype_codes[order]

        changes = np.flatnonzero(np.diff(self._member_hids)) + 1 # find where halo IDs change: +1 accounts for the zeroth case
        self._offsets = np.concatenate(([0], changes)) # where each halo begins
        self._lengths = np.diff(np.concatenate((self._offsets, [len(self._member_hids)]))) # size of halo membership
        self._unique_hids = self._member_hids[self._offsets] 

        max_hid = self._unique_hids.max()
        # halo ID to index: map hid to csr position
        self._hid_to_idx = np.full(max_hid + 1, -1, dtype=np.int32) # array of -1s with len(nhalos)
        self._hid_to_idx[self._unique_hids] = np.arange(len(self._unique_hids)) # set val[i] = unique halo

    def get_halo_particles(self, halo_id, ptype=None):
        """
        Gets the pIDs for a single halo. 
        Can also do specific ptypes if desired.
        """
        # dict lookup to find the index
        idx = self._hid_to_idx[halo_id]
        if idx == -1: # if it lands on a particle
            return np.empty(0, dtype=np.int64) # return nothing
        
        # extract csr format
        start = self._offsets[idx]
        end = start + self._lengths[idx]
        
        # default case: grab all particles (halo readers filter in Octavian particles)
        if ptype is None:
            return self._member_pids[start:end]
        
        # in case you fancy a specific ptype
        mask = self._member_ptypes[start:end] == ptype
        return self._member_pids[start:end][mask]
    
    def get_all_memberships(self, ptype=None):
        """
        Creates flat, aligned (hids, pids) arrays.
        For Caesar-esque progenitor matching.
        """
        if ptype is None:
            return self._member_hids, self._member_pids
        mask = self._member_ptypes == ptype
        return self._member_hids[mask], self._member_pids[mask]
    
    def branch_membership(self, mode='field'):
        """
        Decides on the 'final' membership of a particle (particles can appear in multiple halos)
        
        Field mode: particle belongs to top-level halo (field halo, in AHF paper)
        Subhalo mode: particle belongs to bottom-level halo 
        """
        if mode == 'field':
            resolved_hids = self.tree.field_map[self._member_hids]
        elif mode == 'subhalo':
            resolved_hids = self._member_hids
        else:
            raise ValueError(f"Mode {mode} not supported (yet...)")
        
        return self._deduplicate(
            self._member_pids, self._member_ptypes, 
            resolved_hids, prefer_deepest=(mode == 'subhalo')
        )

    def _deduplicate(self, pids, ptypes, hids, prefer_deepest=False):
        """
        Masks particles to one halo.

        Field mode: order is agnostic
        Subhalo mode: deepest assignment (sub-est halo)
        """
        if prefer_deepest:
            
            sort_key = np.lexsort((-self.tree._depth_lookup[hids], pids)) # https://numpy.org/devdocs/reference/generated/numpy.lexsort.html
        else:
            sort_key = np.argsort(pids)
        
        sorted_pids = pids[sort_key]
        sorted_ptypes = ptypes[sort_key]
        sorted_hids = hids[sort_key]
        
        # locate a pID's first occurrence with boolean masking (array is already sorted)
        mask = np.empty(len(sorted_pids), dtype=bool)
        mask[0] = True # first and last are always unique
        mask[1:] = sorted_pids[1:] != sorted_pids[:-1] # shifting the array one to the left catches the boundaries

        return sorted_pids[mask], sorted_ptypes[mask], sorted_hids[mask]

# optimised numba functions; gets us to C++ speed on these large loops.

@njit
def compute_depths(halo_ids, parent_ids, id_to_idx):
    depths = np.zeros(len(halo_ids), dtype=np.int32)
    for i in range(len(halo_ids)):
        d = 0
        current_parent = parent_ids[i]
        while current_parent != -1:
            idx = id_to_idx[current_parent]
            if idx == -1:
                break
            d += 1
            current_parent = parent_ids[idx]
        depths[i] = d
    return depths

@njit
def build_field_map(halo_ids, parent_ids, id_to_idx):
    n = id_to_idx.shape[0]
    field_map = np.arange(n, dtype=np.int64)
    for i in range(len(halo_ids)):
        hid = halo_ids[i]
        current = hid
        while True:
            idx = id_to_idx[current]
            if idx == -1:
                break
            parent = parent_ids[idx]
            if parent == -1:
                break
            current = parent
        field_map[hid] = current
    return field_map