import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
import yaml
import geopandas as gpd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

with open("../config.yaml") as f:
    config = yaml.safe_load(f)


def check_parameters_quality(parameters: dict):
    if round(sum(parameters['split_manufacturer'].values()), 10) != 1:
        raise ValueError('The sum of split_manufacturer values must be equal to 1')

    if round(sum(parameters['split_station_type'].values()), 10) != 1:
        raise ValueError('The sum of split_station_type values must be equal to 1')

    if not ('small' in parameters['prefered_order_of_station_type'] \
        and 'medium' in parameters['prefered_order_of_station_type'] \
        and 'large' in parameters['prefered_order_of_station_type'] \
        and len(parameters['prefered_order_of_station_type']) == 3) :
        raise ValueError("The prefered_order_of_station_type must contain 'small', 'medium' and 'large' and nothing else")


def compute_quantity_h2(parameters:dict, manufacturers_desc:dict, verbose:bool=True) -> dict[int, dict]:
    """ This function... """
    
    check_parameters_quality(parameters)

    # Compute number of active trucks on the road on a daily basis
    nb_daily_active_trucks = int(parameters['nb_trucks'] * parameters['activation_rate'])

    # Compute number of trucks of each manufacturer on the road on a daily basis
    nb_daily_trucks_man_1 = int(nb_daily_active_trucks * parameters['split_manufacturer']['man_1'])
    nb_daily_trucks_man_2 = int(nb_daily_active_trucks * parameters['split_manufacturer']['man_2'])
    nb_daily_trucks_man_3 = nb_daily_active_trucks - nb_daily_trucks_man_2 - nb_daily_trucks_man_1

    # Get the number of km travelled by 1 truck of each manufacturer on a daily basis
    man_1_nb_daily_km = parameters['avg_daily_km'][parameters['manufacturers_desc']['man_1']['type_of_PL']]
    man_2_nb_daily_km = parameters['avg_daily_km'][parameters['manufacturers_desc']['man_2']['type_of_PL']]
    man_3_nb_daily_km = parameters['avg_daily_km'][parameters['manufacturers_desc']['man_3']['type_of_PL']]

    # Compute number of km travelled by each manufacturer fleet on a daily basis
    nb_daily_km_man_1 = nb_daily_trucks_man_1 * man_1_nb_daily_km
    nb_daily_km_man_2 = nb_daily_trucks_man_2 * man_2_nb_daily_km
    nb_daily_km_man_3 = nb_daily_trucks_man_3 * man_3_nb_daily_km
    nb_daily_km = nb_daily_km_man_1 + nb_daily_km_man_2 + nb_daily_km_man_3

    # Compute the actual autonomy of the trucks of each manufacturer (in km)
    actual_autonomy_man_1 = round(parameters['manufacturers_desc']['man_1']['autonomy'] * (1 - parameters['average_tank_filling_rate_before_refill']))
    actual_autonomy_man_2 = round(parameters['manufacturers_desc']['man_2']['autonomy'] * (1 - parameters['average_tank_filling_rate_before_refill']))
    actual_autonomy_man_3 = round(parameters['manufacturers_desc']['man_3']['autonomy'] * (1 - parameters['average_tank_filling_rate_before_refill']))

    # Compute the number of daily charges of the trucks of each manufacturer
    man_1_nb_daily_refill = round(man_1_nb_daily_km / actual_autonomy_man_1, 2)
    man_2_nb_daily_refill = round(man_2_nb_daily_km / actual_autonomy_man_2, 2)
    man_3_nb_daily_refill = round(man_3_nb_daily_km / actual_autonomy_man_3, 2)

    # Compute the number of daily refill of each manufacturer fleet
    nb_daily_refill_man_1 = round(man_1_nb_daily_refill * nb_daily_trucks_man_1)
    nb_daily_refill_man_2 = round(man_2_nb_daily_refill * nb_daily_trucks_man_2)
    nb_daily_refill_man_3 = round(man_3_nb_daily_refill * nb_daily_trucks_man_3)
    nb_daily_refills = nb_daily_refill_man_1 + nb_daily_refill_man_2 + nb_daily_refill_man_3

    # Compute the daily quantity of h2 for each manufacturer fleet
    quantity_h2_consumed_man_1 = nb_daily_refill_man_1 * manufacturers_desc['man_1']['tank_size']
    quantity_h2_consumed_man_2 = nb_daily_refill_man_1 * manufacturers_desc['man_2']['tank_size']
    quantity_h2_consumed_man_3 = nb_daily_refill_man_1 * manufacturers_desc['man_3']['tank_size']
    quantity_h2_consumed = quantity_h2_consumed_man_1 + quantity_h2_consumed_man_2 + quantity_h2_consumed_man_3

    # Adjust the daily quantity of h2 needed
    quantity_h2_required = quantity_h2_consumed * (1 + parameters['security_buffer'])
    quantity_h2 = int(round(quantity_h2_required * (parameters['strategic_positioning_index'])))
    
    # Print logs and intermediate results
    if verbose:
        print(f"Estimated number of H2 trucks in {parameters['year']}: {parameters['nb_trucks']}")
        print(f"\nNumber of daily active H2 trucks (activation_rate: {parameters['activation_rate']})")   
        print(f"  - Man.1 ({manufacturers_desc['man_1']['name']}): {nb_daily_trucks_man_1} ({round(parameters['split_manufacturer']['man_1'], 2) * 100}%)")
        print(f"  - Man.2 ({manufacturers_desc['man_2']['name']}): {nb_daily_trucks_man_2} ({round(parameters['split_manufacturer']['man_2'], 2) * 100}%)")
        print(f"  - Man.3 ({manufacturers_desc['man_3']['name']}): {nb_daily_trucks_man_3} ({round(parameters['split_manufacturer']['man_3'], 2) * 100}%)")
        print(f"  - Total: {nb_daily_active_trucks}")
        print(f"\nDistance travelled daily based on the type of PL (in km, short-distance: {parameters['avg_daily_km']['short-distance']}; long-distance: {parameters['avg_daily_km']['long-distance']})")
        print(f"  - Man.1: {nb_daily_km_man_1}")
        print(f"  - Man.2: {nb_daily_km_man_2}")
        print(f"  - Man.3: {nb_daily_km_man_3}")
        print(f"  - Total: {nb_daily_km}")
        print(f"\nActualised autonomy of a truck (in km, average_tank_filling_rate_before_refill = {parameters['average_tank_filling_rate_before_refill']})")
        print(f"  - Man.1: {actual_autonomy_man_1}")
        print(f"  - Man.2: {actual_autonomy_man_2}")
        print(f"  - Man.3: {actual_autonomy_man_3}")
        print(f"\nNumber of necessary daily charges for a truck:")
        print(f"  - Man.1: {man_1_nb_daily_refill}")
        print(f"  - Man.2: {man_2_nb_daily_refill}")
        print(f"  - Man.3: {man_3_nb_daily_refill}")
        print(f"\nDaily consumed quantity of H2 (in kg)")
        print(f"  - Man.1: {quantity_h2_consumed_man_1} (tank_size: {manufacturers_desc['man_1']['tank_size']})")
        print(f"  - Man.2: {quantity_h2_consumed_man_2} (tank_size: {manufacturers_desc['man_2']['tank_size']})")
        print(f"  - Man.3: {quantity_h2_consumed_man_3} (tank_size: {manufacturers_desc['man_3']['tank_size']})")
        print(f"  - Total: {quantity_h2_consumed}")
        print(f"\nDaily required quantity of H2 (in kg, with a security buffer of {round(parameters['security_buffer'] * 100)}%)")
        print(f"  - Total: {quantity_h2_required}")
        print(f"\nDaily budgeted quantity of H2 (in kg, with a strategic positioning index of {round(parameters['strategic_positioning_index'])})")
        print(f"  - Total: {quantity_h2}")

    return quantity_h2


def compute_number_of_stations(quantity_h2: int, parameters: dict, stations_desc: dict) -> int:
    """ This function... """
    check_parameters_quality(parameters)
    return int(round(quantity_h2 / (parameters['split_station_type']['small'] * stations_desc['small']['storage_onsite'] \
                                    + parameters['split_station_type']['medium'] * stations_desc['medium']['storage_onsite'] \
                                    + parameters['split_station_type']['large'] * stations_desc['large']['storage_onsite']) + 0.5))


def perform_best_stations_split(quantity_h2: int, parameters: dict, stations_desc: dict, verbose:bool=True) -> dict:
    """ This function... """
    check_parameters_quality(parameters)
    nb_stations = compute_number_of_stations(quantity_h2, parameters, stations_desc)

    # Compute first estimation (rouded up)
    nb_stations_small = nb_stations * parameters['split_station_type']['small']
    nb_stations_medium = nb_stations * parameters['split_station_type']['medium']
    nb_stations_large = nb_stations * parameters['split_station_type']['large']
    proposed_strategy = {'small': round(nb_stations_small + 0.5), 
                        'medium': round(nb_stations_medium + 0.5), 
                        'large': round(nb_stations_large + 0.5)}
    proposed_quantity_h2 = proposed_strategy['small'] * stations_desc['small']['storage_onsite'] \
                            + proposed_strategy['medium'] * stations_desc['medium']['storage_onsite'] \
                            + proposed_strategy['large'] * stations_desc['large']['storage_onsite']
    
    # Print logs and intermediate results
    if verbose:
        print(f"\nEstimated number of stations needed")
        print(f"  - Small: {nb_stations_small}")
        print(f"  - Medium: {nb_stations_medium}")
        print(f"  - Large: {nb_stations_large}")
        print(f"  - Total: {nb_stations}")
        print(f"\nFirst proposition - number of stations")
        print(f"  - Small: {proposed_strategy['small']}")
        print(f"  - Medium: {proposed_strategy['medium']}")
        print(f"  - Large: {proposed_strategy['large']}")
        print(f"First proposition - quantity of H2: {proposed_quantity_h2}")
        print("\n> Performing optimization...")

    # Compute best estimation
    for _ in range(5): # may have to delete the smallest stations several times without being able to delete a medium one (depends on the stations config)
        for station_type in parameters['prefered_order_of_station_type'][::-1]:
            if proposed_quantity_h2 - quantity_h2 > stations_desc[station_type]['storage_onsite']:
                proposed_strategy[station_type] -= 1
                proposed_quantity_h2 = proposed_strategy['small'] * stations_desc['small']['storage_onsite'] \
                                    + proposed_strategy['medium'] * stations_desc['medium']['storage_onsite'] \
                                    + proposed_strategy['large'] * stations_desc['large']['storage_onsite']
    
    # Print logs and intermediate results            
    if verbose:
        print("> Optimization done!")
        print(f"\nFinal number of stations")
        print(f"  - Small: {proposed_strategy['small']}")
        print(f"  - Medium: {proposed_strategy['medium']}")
        print(f"  - Large: {proposed_strategy['large']}")
        print(f"\nQuantity of H2 to reach: {round(quantity_h2)}")
        print(f"Proposed quantity of H2: {proposed_quantity_h2}")

    proposed_strategy['total'] = proposed_strategy['small'] + proposed_strategy['medium'] + proposed_strategy['large']
    proposed_strategy['quantity_h2_to_reach'] = round(quantity_h2)
    proposed_strategy['quantity_h2_proposed'] = proposed_quantity_h2
    return proposed_strategy


def define_national_strategies(quantity_h2:int, parameters:dict, stations_desc:dict, verbose:bool=False) -> pd.DataFrame:
    """ This function... """
    national_strategy = perform_best_stations_split(quantity_h2, parameters=parameters, stations_desc=stations_desc, verbose=verbose)
    national_strategy['region'] = 'Total'
    return pd.DataFrame(national_strategy, index=[0])[['region', 'quantity_h2_to_reach', 'quantity_h2_proposed', 'small', 'medium', 'large', 'total']]
    

def define_regional_strategies(quantity_h2:int, region_breakdown: dict, parameters:dict, stations_desc:dict, verbose:bool=False) -> pd.DataFrame:
    """ This function... """
    regional_strategies_list = []
    for region, prop in region_breakdown.items():
        regional_quantity_h2 = quantity_h2 * prop
        regional_strategy = perform_best_stations_split(regional_quantity_h2, parameters=parameters, stations_desc=stations_desc, verbose=verbose)
        regional_strategy['region'] = region
        regional_strategies_list.append(regional_strategy)
    regional_strategies = pd.DataFrame(regional_strategies_list)[['region', 'quantity_h2_to_reach', 'quantity_h2_proposed', 'small', 'medium', 'large', 'total']]
    return regional_strategies.append(regional_strategies.sum(numeric_only=True), ignore_index=True).fillna('Total')


def define_best_regional_strategies(parameters:dict, manufacturers_desc:dict, stations_desc:dict, region_breakdown:dict, verbose:bool=False)-> pd.DataFrame:
    """ This function... """
    quantity_h2 = compute_quantity_h2(parameters=parameters, manufacturers_desc=manufacturers_desc, verbose=verbose)
    return define_regional_strategies(quantity_h2, region_breakdown, parameters, stations_desc, verbose=False)


def define_best_national_strategy(parameters:dict, manufacturers_desc:dict, stations_desc:dict, verbose:bool=False)-> pd.DataFrame:
    """ This function... """
    quantity_h2 = compute_quantity_h2(parameters=parameters, manufacturers_desc=manufacturers_desc, verbose=verbose)
    return define_national_strategies(quantity_h2=quantity_h2, parameters=parameters, stations_desc=stations_desc, verbose=verbose)