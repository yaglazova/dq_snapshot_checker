import numpy as np
import pandas as pd
from pyspark.sql.dataframe import DataFrame


def sub_psi(e_perc: list, a_perc: list):
    """ Function calculates sub psi for specified column """
    if a_perc == 0:
        a_perc = 0.0001
    if e_perc == 0:
        e_perc = 0.0001
        
    value = (e_perc - a_perc) * np.log(e_perc / a_perc)
    return value


def psi(expected_percents: list, actual_percents: list, breakpoints: list):
    """ Function calculates psi for specified column """
    psi_dict = {}
    for i in range(0, len(expected_percents)):
        psi_val = sub_psi(expected_percents[i], actual_percents[i])
        if breakpoints[i] == 'None':
            psi_dict.update({'empty_value': psi_val})
        else:
            psi_dict.update({breakpoints[i]: psi_val})
    psi_value = np.sum(list(psi_dict.values()))
    psi_dict = {k:v for k,v in sorted(psi_dict.items(), key=lambda x: x[1], reverse=True)}
    return psi_value, psi_dict


def atr_check(expected: DataFrame, actual: DataFrame, join_col, atr_col: list, round_n=1) ->pd.DataFrame:
    """
    Function performs columns quality check by attribute

    Arguments:
    expected - expected dataset
    actual - actual dataset, 
    join_col - list of columns or single column, list or str
    atr_col - list of columns to check, list
    round_n - precision for numeric values, int
    return: report, pd.DataFrame
    """
    
    report_data = pd.DataFrame()
    if type(join_col) == list:
        filt = [i for i in join_col]
    else:
        filt = [join_col]
        
    num_cols = [col for col in expected.select(filt + atr_col).columns
                if expected.select(col).dtypes[0][1] in ['float', 'decimal', 'double']]
    if len(num_cols) > 0:
        for col in num_cols:
            expected = expected.withColumn(col, F.round(col, round_n))
            actual = actual.withColumn(col, F.round(col, round_n))
    for atr in atr_col:
        a = expected.select(filt+[atr]).dropna()
        b = actual.select(filt+[atr]).dropna()
        intersect = a.alias('t1').join(b.alias('t2'), on=join_col, how='inner').filter(f't1.{atr}==t2.{atr}')
        len_a = a.count()
        len_b = b.count()
        intersect_count = intersect.count()
        completness_b = round(100 * intersect_count / len_b, 3)
        atr_data = pd.DataFrame({'join_column': f'{join_col}', 'atrib_column': atr, 'expected_row_n': len_a,
                                 'actual_row_n': len_b, 'intersect_n': intersect_count, 'completeness': completness_b},
                                index=[1])
        report_data = pd.concat([report_data, atr_data])
    report_data = report_data.reset_index(drop=True)
    return report_data
