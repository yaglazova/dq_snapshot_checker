import pandas as pd
from tqdm import tqdm

from .snapshot_checker import *


class RunSnapshotChecker:
    """
    Class returns quality report depends on dataset type.
    Arguments:
            expected - expected dataset, Spark or Pandas dataframe,
            actual -  actual dataset, Spark or Pandas dataframe,
            plot - plot column distribution, by default is False, bool
            category_grp - specify if grouping category columns is necessary, True by default, bool
            psi_th - sensitivity threshold:by default 0.2, float
    Returns:
            df -  quality report, pd.DataFrame

    ---------------------------
    Contact for bug reports or questions: 
    Yana Glazova 
    e-mail: yana_glazova@vk.com
    """
    def __init__(self, expected, actual, plot=False, category_grp=True, psi_th=0.2):

        self.expected = expected
        self.actual = actual
        self.ds_type = 'pandas' if type(expected) == pd.DataFrame else 'spark'
        self.category_grp = category_grp
        self.plot = plot
        self.psi_th = psi_th

    def get_report(self) -> pd.DataFrame:
        """
        Report warp for PSI quality check. Performs quality check for dataset.
        Columns description:   
            column - checked column
            anomaly_score - anomaly score depeends on estimated metric
            metric_name - name of metric used
            check_result - model check result with respect of basic metric tresholds (if > 0.2 then anomaly) 
            failed_bucket - top5 of buckets with highest PSI (if applicable)
            new_category -  name of categories appeared in actual dataset (only for categorical data otherwise empty)
            disappeared_category -  name of categories absent in actual dataset
                (only for categorical data otherwise empty)
        """
        assert len(self.expected.columns) == len(self.actual.columns)
        
        data_cols = self.expected.columns
        score_dict = {}
        df = pd.DataFrame()
        new_cat_dict = {}
        for col in tqdm(data_cols):
            if self.ds_type == 'spark':
                a = self.expected.select(col)
                b = self.actual.select(col)
                psi_res = SparkPSI(a, b, col, plot=self.plot, category_grp=self.category_grp)
            else:
                psi_res = PandasPSI(self.expected, self.actual, col, plot=self.plot)

            score, psi_dict, new_cats, abs_cats = psi_res.psi_result()

            if len(new_cats) > 0:
                new_cat_dict.update({col:new_cats})
            score_dict.update({col: score})
            check_result = 'OK' if score < self.psi_th else 'NOK'

            failed_buckets = list(psi_dict.keys())[:5] if score > self.psi_th else []
            if 'metric' in psi_dict:
                new_cats = []
                abs_cats = []
                metric_name = psi_dict['metric']
                if metric_name == 'unique_index':
                    failed_buckets = None
            else:
                metric_name = 'stability_index'
            df_tmp = pd.DataFrame({'column': col, 'anomaly_score': score, 'metric_name': metric_name,
                                   'check_result': check_result, 'failed_bucket': f'{failed_buckets}',
                                   'new_category': f'{new_cats}', 'disappeared_category': f'{abs_cats}'}, index=[1])
            df = pd.concat([df, df_tmp])  
        df = df.reset_index(drop=True)
        return df

    def atribution_check(self, join_col, atr_col: list, round_n=1) ->pd.DataFrame:
        """
        Function performs columns quality check by attribute. Only for spark.

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
            
        num_cols = [col for col in self.expected.select(filt+atr_col).columns
                    if self.expected.select(col).dtypes[0][1] in ['float', 'decimal', 'double']]
        if len(num_cols) > 0:
            for col in num_cols:
                self.expected = self.expected.withColumn(col, F.round(col, round_n))
                self.actual = self.actual.withColumn(col, F.round(col, round_n))
        for atr in tqdm(atr_col):
            a = self.expected.select(filt+[atr]).dropna()
            b = self.actual.select(filt+[atr]).dropna()
            intersect = a.alias('t1').join(b.alias('t2'), on=join_col, how='inner').filter(f't1.{atr}==t2.{atr}')
            len_a = a.count()
            len_b = b.count()
            intersect_count = intersect.count()
            completness_b = round(100 * intersect_count / len_b, 3)
            atr_data = pd.DataFrame({'join_column': f'{join_col}', 'atrib_column': atr, 'expected_row_n': len_a,
                                     'actual_row_n': len_b, 'intersect_n': intersect_count, 
                                     'completeness': completness_b}, index=[1])
            report_data = pd.concat([report_data, atr_data])
        report_data = report_data.reset_index(drop=True)
        return report_data





        


