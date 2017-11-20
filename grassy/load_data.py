from __future__ import print_function, absolute_import, division
from builtins import *

import os
import pandas as pd
def _get_path(rel_path):
    """get absolute path from path relative to package"""
    this_dir, this_file = os.path.split(os.path.abspath(__file__))
    return os.path.join(this_dir, rel_path)

# load data:
def _get_data():
    # alias:
    df_alias = pd.read_csv(_get_path('data/alias.csv.gz'), index_col=0).reset_index()
    df_srm = pd.read_csv(_get_path('data/srm_values.csv'), index_col=0)
    df_cert = pd.read_csv(_get_path('data/certlist.csv'), index_col=0)

    return df_alias, df_srm, df_cert

_df_alias, _df_srm, _df_cert = _get_data()

