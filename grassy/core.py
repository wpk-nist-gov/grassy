

from __future__ import print_function, absolute_import, division
from builtins import *

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms


from .cached_decorators import cached_clear, cached

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



class _Grass(object):

    def __init__(self, data, srm, alias=None, srm_values=None, cert=None):
        # set frames
        self._df_alias = alias or _df_alias
        self._df_srm_values = srm_values or _df_srm
        self._df_cert = cert or _df_cert

        # set srm
        self.srm = srm

        # set data
        self.data = data


    @property
    def srm(self):
        return self._srm

    @srm.setter
    @cached_clear()
    def srm(self, val):
        srms = self._df_srm_values['SRM'].unique()
        if val not in srms:
            raise ValueError('srm must be in %s'%srms)
        self._srm = val


    @property
    def data(self):
        return self._data

    @data.setter
    @cached_clear()
    def data(self, val):
        assert isinstance(val, pd.DataFrame)
        self._data = val.assign(ID=lambda x: np.arange(len(x)))


    @property
    def data_agg(self):
        """aggregated stats on data"""
        return (
            self.data
            .pipe(pd.melt, id_vars=['Name','ID'])
            .groupby(['Name','ID'],as_index=False)['value']
            .agg(['mean','std'])
            .sort_index(level='ID')
            .reset_index()
        )


    @property
    @cached()
    def srm_values(self):
        """limited set of srm values"""
        val = self.srm
        return (
            self._df_srm_values
            .query('SRM == @val')
            [['SRM','Measureand','InChI','CertRef', 'Value', 'Uncertainty',
              'Mass/Volume', 'UnitNumerator', 'UnitDenominator','WetDry']]
        )


    @property
    @cached()
    def merge_name(self):
        """merge on name/Measureand"""
        return (
            self.data_agg
            .merge(self.srm_values,
                   left_on='Name', right_on='Measureand', how='left')
            .dropna(subset=['Measureand'])
        )


    @property
    @cached()
    def merge_inchi(self):
        """merge InChi[name]"""
        return (
            self.data_agg
            .merge(self._df_alias, left_on='Name', right_on='Alias', how='left')
            .dropna(subset=['InChI'])
            .merge(self.srm_values,
                   left_on='InChI', right_on='InChI', how='left')
            .dropna(subset=['Measureand'])
        )

    @property
    @cached()
    def merge_final(self):
        """join merge_name and merge_inchi and drop duplicates"""
        return (
            pd.concat((self.merge_name, self.merge_inchi))
            # order
            [self.merge_name.columns]
            # remove duplicates
            .drop_duplicates()
            # sort
            .sort_values('ID')
        )

    @property
    @cached()
    def duplicates(self):
        m = self.merge_final['Name'].duplicated()
        v = self.merge_final.loc[m, 'Name'].values
        return self.merge_final.query('Name in @v')


    def plot_control(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10,3))

        t = (
            self.merge_final
            .assign(y=lambda x: (x['mean'] - x['Value'])/(x['Uncertainty']/2.0))
            .assign(dy = lambda x: x['std']/x['Uncertainty']*2.0)

        )


        ax.axhline(y=0, color='k')
        ax.axhline(y=-3.0, color='r', ls=':')
        ax.axhline(y=+3.0, color='r', ls=':')
        trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.fill_between([0, 1], [-2.0]*2, [+2.0]*2, transform=trans, color='g', alpha=0.25)

        ax.errorbar(x=np.arange(len(t)), y=t['y'], yerr=t['dy'], linestyle='None', marker='o')
        _ = ax.set_xticks(np.arange(len(t)))
        _ = ax.set_xticklabels(t['Name'], rotation=90)

        ax.set_xlabel('Analyte')
        ax.set_ylabel('Coverage Equivalent (k-eq)')
        return ax

    

    @classmethod
    def from_csv(cls, path, srm, read_kws=None, **kwargs):
        read_kws = read_kws or {}
        df = pd.read_csv(path, **read_kws)
        return cls(df, srm=srm, **kwargs)

    def summary(self, plt_kws=None):
        import datetime, os, pwd
        from IPython.display import display, Markdown

        header = _header_template.format(
            date=str(datetime.datetime.now()),
            user=pwd.getpwuid(os.getuid())[0],
            srm=self.srm,
            srm_num=self.srm.split()[-1]
        )

        disp = lambda x: display(Markdown(x))
        disp(header)

        plt_kws = plt_kws or {}
        self.plot_control(**plt_kws)
        plt.show()

        legend = _legend_template.format(srm=self.srm)
        disp(legend)

        disp("## SRM Accuracy summary")
        display(self.merge_final)

        disp("## Duplicates")
        display(self.duplicates)



_header_template="""# Feature reduction assistant for metabalomics
### NIST Marine ESB Data Tool Development
### FRAMey v0.1: last update November 2017

---

* Timestamp: {date}
* User: {user}

* <a href="https://www-s.nist.gov/srmors/certificates/view_certPDF.cfm?certificate={srm_num}" target="_blank">{srm}</a>
---
"""

_legend_template = """**Figure 1:** Accuracy assessment of batch XXXXX for {srm}.  Values are presented as normalized coverage equivalents at the mean (dots) and 1sd (error bars) of measurements, overlaid onto the certificate value (blue line) and uncertainty (green~95% coverage, red~99% coverage).

---
"""
