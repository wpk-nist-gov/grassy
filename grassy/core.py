

from __future__ import print_function, absolute_import, division
from builtins import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

from fuzzywuzzy import process

from .cached_decorators import cached_clear, cached

from .load_data import _df_alias, _df_srm, _df_cert

## utility functinos
def _drop_vals(df, vals, key='ID'):
    return df.loc[~df[key].isin(vals)]


def _reorder_columns(df, first):
    """
    return df[cols_new], where cols_new is cols, but ordered such that
    first_cols are first
    """

    cols = df.columns
    cols_new = list(first) + list(cols.drop(first))
    return df[cols_new]


class Grass(object):

    def __init__(self, data_raw, srm_name, alias=None, srm_values=None):
        """
        Parameters
        ----------
        data_raw : pd.DataFrame
            data_raw data
        srm_name : str
            name of srm
        alias : pd.DataFrame, optional
            frame with alias info. 
        srm_values : pd.DataFrame, optional
            frame with srm data.
        """
        # set frames
        self._df_alias = alias or _df_alias
        self._df_srm_values = srm_values or _df_srm

        # set srm
        self.srm_name = srm_name

        # set data_raw
        self.data_raw = data_raw


    @property
    def srm_name(self):
        return self._srm_name

    @srm_name.setter
    @cached_clear()
    def srm_name(self, val):
        srms = self._df_srm_values['SRM'].unique()
        if val not in srms:
            raise ValueError('srm must be in %s'%srms)
        self._srm_name = val


    @property
    def data_raw(self):
        return self._data_raw

    @data_raw.setter
    @cached_clear()
    def data_raw(self, val):
        assert isinstance(val, pd.DataFrame)
        self._data_raw = val.assign(ID=lambda x: np.arange(len(x)))


    @property
    def data_agg(self):
        """aggregated stats on data"""
        return (
            self.data_raw
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
        val = self.srm_name
        return (
            self._df_srm_values
            .query('SRM == @val')
            [['SRM','Measureand','InChI','CertRef', 'Value', 'Uncertainty','u',
              'Mass/Volume', 'UnitNumerator', 'UnitDenominator','WetDry']]
            # add in row ID
            .assign(row_ID=lambda x: np.arange(len(x)))
        )


    # some srm_valus with merged in stuff
    @property
    @cached()
    def _srm_alias_inchi(self):
        """merge srm with alias on inchi"""
        return (
            self.srm_values.dropna(subset=['InChI'])
            .merge(self._df_alias.dropna(subset=['InChI']), how='inner', on='InChI')
        )

    @property
    @cached()
    def _srm_alias_analyte(self):
        """
        merge srm with alias on Measureand / Analyte
        """
        return (
            self.srm_values.dropna(subset=['Measureand'])
            .merge(self._df_alias.dropna(subset=['Analyte']),
                   how='inner', left_on='Measureand', right_on='Analyte',
                   suffixes=['', '_alias']
            )
        )



    ################################################################################
    # exact matching
    @property
    @cached()
    def _merge_agg_name_measureand(self):
        """
        merge data_agg with srm_values on name/Measureand
        """
        return (
            self.data_agg
            .merge(self.srm_values,
                   left_on='Name', right_on='Measureand', how='inner')
            .assign(match_method='Name/Measureand')
            .assign(match_score=100)
            # add in Alias, Analyte
            .assign(Alias=lambda x: x['Measureand'],
                    Analyte=lambda x: x['Measureand'])
        )


    @property
    @cached()
    def _merge_agg_name_alias_inchi(self):
        # remove those rows that have already been matched above
        # ids = self._merge_agg_name_measureand['ID'].unique()
        # df = _drop_vals(self.data_agg, ids, key='ID')
        df = self.data_agg
        return (
            df
            .merge(self._srm_alias_inchi,
                   left_on='Name', right_on='Alias', how='inner'
            )
            .assign(match_method='InchI/Alias')
            .assign(match_score=100)
        )

    @property
    def _first_cols(self):
        return ['ID', 'Name','Measureand','Alias','Analyte',
                'match_method','match_score',
                'mean','std','Value','Uncertainty','u']


    @property
    @cached()
    def _merge_final(self):
        """join merge_name and merge_inchi and drop duplicates"""
        df =  (
            pd.concat((self._merge_agg_name_measureand, self._merge_agg_name_alias_inchi))
            # order
            # [self._merge_agg_name_measureand.columns]
            # remove duplicates
            # sort
            .sort_values('ID')

            # order
            .pipe(_reorder_columns, self._first_cols)
        )
        return df

    @property
    @cached()
    def merge_final(self):
        # remove duplicates
        df = self._merge_final.reset_index(drop=True)
        idx = df.drop(['Alias','Analyte', 'match_method',
                       'match_score'], axis=1).drop_duplicates().index
        return df.loc[idx].reset_index(drop=True)        


    ###############################/#################################################
    # approximate matching
    @property
    @cached()
    def _data_agg_nomatch(self):
        id_found_match = self._merge_final['ID'].values
        return self.data_agg.drop(id_found_match)

    @staticmethod
    def _score_values(values, reference,
                      val_name='Name', new_name='Name_match', score_name='match_score'):
        """for list of values, get score agains reference"""
        L = []
        for val in values:
            new_val, score = process.extractOne(val, reference)
            L.append((val, new_val, score))
        return pd.DataFrame(L, columns=[val_name, new_name, score_name])


    def _get_scored_frame(self, data, ref, data_key='Name', ref_key='Measureand'):
        """socre across frame"""
        match = self._score_values(
            data[data_key],
            ref[ref_key].unique(),
            val_name=data_key,
            new_name=ref_key
        )

        return (
            match
            .merge(data, on=data_key, how='inner')
            .merge(ref, on=ref_key, how='inner')
        )

    @property
    @cached()
    def _score_name_measureand(self):
        df = _drop_vals(self.data_agg, self._merge_final['ID'].unique())
        return (
            self._get_scored_frame(
                data=df,
                ref=self.srm_values, ref_key='Measureand')
            .assign(match_method='Name/Measureand-score')
            # add alias, Measureand
            .assign(Alias=lambda x: x['Measureand'], Analyte=lambda x: x['Measureand'])
        )

    @property
    @cached()
    def _score_name_alias_inchi(self):
        df = _drop_vals(self.data_agg, self._merge_final['ID'].unique())
        return (
            self._get_scored_frame(
                data=df,
                ref=self._srm_alias_inchi, ref_key='Alias')
            .assign(match_method='InChI/Alias-score')
        )

    @property
    @cached()
    def _score_final(self):
        df = (
            pd.concat((self._score_name_measureand, self._score_name_alias_inchi))
            # sort
            .sort_values('ID')

            # order
            .pipe(_reorder_columns, self._first_cols)
        )
        return df


    @property
    @cached()
    def score_final(self):
        # remove duplicates
        df = self._score_final.reset_index(drop=True)
        idx = (
            df.drop(['Alias','Analyte','match_method', 'match_score'], axis=1)
            .drop_duplicates()
            .index
        )
        return df.loc[idx].reset_index(drop=True)




    def get_final(self, score_min=100):
        if score_min >= 100:
            df = self.merge_final.reset_index(drop=True)

        else:
            df = (
                self.merge_final
                .append(
                    self.score_final.query('match_score >= @score_min')
                )
                .sort_values('ID')
                .reset_index(drop=True)
            )

        return df


    def _get_duplicates(self, df):
        m = df['Name'].duplicated()
        v = df.loc[m, 'Name'].values
        return df.query('Name in @v')


    def plot_control(self, df, ax=None,):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10,3))

        t = (
            df
            .assign(y=lambda x: (x['mean'] - x['Value'])/(x['Uncertainty']/2.0))
            .assign(dy = lambda x: x['std']/x['Uncertainty']*2.0)

        )

        for match_method in ['Name/Measureand', 'InchI/Alias',
                             'Name/Measureand-score', 'InChI/Alias-score']:

            tt = t.query('match_method==@match_method')
            x = tt.index.values
            y = tt['y'].values
            yerr = tt['dy'].values
            # bug in errobar leads to legend issues if len(x) ==0,
            # so plot data, then plot error bars
            l, = ax.plot(x, y, marker='o', linestyle='None', label=match_method)
            if len(x) > 0:
                ax.errorbar(x=x, y=y, yerr=yerr,
                            linestyle='None', marker='None', color=l.get_color())

        ax.legend(title='match method')
        _ = ax.set_xticks(np.arange(len(t)))
        _ = ax.set_xticklabels(t['Name'], rotation=90)


        # add control lines
        ax.axhline(y=0, color='k')
        ax.axhline(y=-3.0, color='r', ls=':')
        ax.axhline(y=+3.0, color='r', ls=':')
        trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.fill_between([0, 1], [-2.0]*2, [+2.0]*2, transform=trans, color='g', alpha=0.25)
        ax.set_xlabel('Analyte')
        ax.set_ylabel('Coverage Equivalent (k-eq)')

        return ax

    @classmethod
    def from_csv(cls, path, srm_name, read_kws=None, **kwargs):
        read_kws = read_kws or {}
        df = pd.read_csv(path, **read_kws)
        return cls(df, srm_name=srm_name, **kwargs)

    def summary(self, plt_kws=None, score_min=95):
        import datetime, os, pwd
        from IPython.display import display, Markdown

        header = _header_template.format(
            date=str(datetime.datetime.now()),
            user=pwd.getpwuid(os.getuid())[0],
            srm_name=self.srm_name,
            srm_num=self.srm_name.split()[-1]
        )

        disp = lambda x: display(Markdown(x))
        disp(header)

        plt_kws = plt_kws or {}

        df = self.get_final(score_min=score_min)

        ax = self.plot_control(df=df, **plt_kws)
        plt.show()

        legend = _legend_template.format(srm_name=self.srm_name)
        disp(legend)

        disp("## SRM Accuracy summary")
        display(df)

        disp("## Duplicates")
        display(self._get_duplicates(df))



_header_template="""# Feature reduction assistant for metabalomics
### NIST Marine ESB Data Tool Development
### FRAMey 0.0.1: last update November 2017

---

* Timestamp: {date}
* User: {user}

* <a href="https://www-s.nist.gov/srmors/certificates/view_certPDF.cfm?certificate={srm_num}" target="_blank">{srm_name}</a>
---
"""

_legend_template = """**Figure 1:** Accuracy assessment of batch XXXXX for {srm_name}.  Values are presented as normalized coverage equivalents at the mean (dots) and 1sd (error bars) of measurements, overlaid onto the certificate value (blue line) and uncertainty (green~95% coverage, red~99% coverage).

---
"""
