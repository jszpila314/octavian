from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from octavian.data_manager import DataManager

import numpy as np
import pandas as pd


def get_group_particle_indexes(data_manager: DataManager, group_name: str) -> None:
  config = data_manager.config

  group_data = data_manager.group_data[group_name]
  groupID = config['groupIDs'][group_name]

  group_props_columns = ['HaloID', 'GalID', 'particle_index']
  for ptype in config['ptypes']:
    data = data_manager.data[ptype][group_props_columns].copy()

    if group_name == 'galaxies':
      data = data.loc[data['GalID'] != -1]

    ptype_list = config['ptype_lists'][ptype]
    
    group_ids = list(group_data.index)
    ptype_particle_lists = []
    for id in group_ids:
      ptype_particle_lists.append(np.array(data.loc[data[groupID] == id, 'particle_index']))

    group_data[ptype_list] = ptype_particle_lists


def get_particle_lists(data_manager: DataManager) -> None:
  config = data_manager.config
  for ptype in config['ptypes']:
    data_manager.load_property('particle_index', ptype)

  groups = config['groups']

  for group in groups:
    get_group_particle_indexes(data_manager, group)
