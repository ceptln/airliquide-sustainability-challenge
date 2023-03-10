import warnings

import pandas as pd
try:
    from pandas.core.common import SettingWithCopyWarning
except ImportError:
    from pandas.errors import SettingWithCopyWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
pd.set_option('display.max_columns', 500)


def clean_traffic_df(traffic:pd.DataFrame)-> pd.DataFrame:
    """ This function... """
    traffic.drop(index=traffic[traffic['pctPL']==0.0].index, inplace=True)
    traffic[traffic.pctPL > 40]['pctPL'] = traffic[traffic.pctPL > 40]['pctPL'] / 10
    traffic.drop(index=traffic[traffic['tmja']==0.0].index, inplace=True)
    traffic = traffic.loc[:, ['route', 'depPrD', 'tmja', 'pctPL', 'longueur', 'geometry']]
    traffic['nb_PL'] = traffic['tmja'] * (traffic['pctPL'] / 100)
    return traffic


def aggregate_traffic_by_dep(traffic:pd.DataFrame, agg_params:dict=None)-> pd.DataFrame:
    """ This function... """
    if agg_params is None:
        agg_params = {
            'depPrD': ['count'],
            'longueur': ['sum'],
            'nb_PL': ['sum']
        }
    traffic_by_dep = traffic.groupby('depPrD').agg(agg_params)
    traffic_by_dep.reset_index(inplace=True)
    traffic_by_dep.columns = ['num_dep', 'nb_routes', 'total_longueur', 'nb_PL']
    return traffic_by_dep


def aggregate_traffic_by_reg(traffic_by_dep:pd.DataFrame, depreg:pd.DataFrame, agg_params:dict=None)-> pd.DataFrame:
    """ This function... """
    traffic_by_reg_prep = pd.merge(depreg, traffic_by_dep, on='num_dep')
    if agg_params is None:
        agg_params = {
            'num_dep': ['count'],
            'nb_routes': ['sum'],
            'total_longueur': ['sum'],
            'nb_PL': ['sum']
        }
    traffic_by_reg = traffic_by_reg_prep.groupby('region_name').agg(agg_params)
    traffic_by_reg.reset_index(inplace=True)
    traffic_by_reg.columns = ['region', 'nb_dep', 'nb_routes', 'total_longueur', 'nb_PL']
    traffic_by_reg['traffic_proportion'] = (traffic_by_reg.nb_PL / traffic_by_reg.nb_PL.sum()).round(5)
    return traffic_by_reg


def create_region_breakdown_dictionary(traffic_by_reg:pd.DataFrame) -> dict:
    """ This function... """
    return traffic_by_reg[['region', 'traffic_proportion']].set_index('region').to_dict()['traffic_proportion']


def create_region_breakdown(traffic:pd.DataFrame, depreg:pd.DataFrame)-> dict:
    """ This function... """
    traffic = clean_traffic_df(traffic)
    traffic_by_dep = aggregate_traffic_by_dep(traffic)
    traffic_by_reg = aggregate_traffic_by_reg(traffic_by_dep, depreg)
    return create_region_breakdown_dictionary(traffic_by_reg)
