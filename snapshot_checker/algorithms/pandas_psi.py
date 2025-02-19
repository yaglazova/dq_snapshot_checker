import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from snapshot_checker.algorithms.utils import psi, sub_psi


class PandasPSI:
    """
    Class calculates population stability index for pandas dataframe. Numeric data split on 10 buckets
    (except null values).
     For categorial data: if n < 20 - each category, 20 < n  100 categories split on groups depends on its size, 
     if  n > 100 - calculates unique_index according to  Jaccard metric. Also class checks good data/null balance.

        arguments:
        expected - expected dataset, Spark dataframe,
        actual -  actual dataset, Spark dataframe,
        column_name - column name, str,
        plot - plot feature distribution, if True. By default - False, bool.
        returns: 
        psi_value - PSI for selected column, float,
        psi_dict - sub psi for each bucket, dict, 
        new_cats - new categories apeared in actual dataset (return empty list for numeric values), list,
        abs_cats - new categories apeared in actual dataset (empty list for numeric values), list
        category_grp - specify if grouping categorry columns is neccessary, True by default, bool
        """
    
    def __init__(self, expected, actual, column_name, plot=False):
        self.expected = expected[column_name].values
        self.actual = actual[column_name].values
        self.actual_len = len(self.actual)
        self.expected_len = len(self.expected)
        self.column_name = column_name
        self.column_type = self.expected.dtype
        self.expected_shape = self.expected.shape
        self.expected_nulls =  np.sum(pd.isna(self.expected))
        self.actual_nulls = np.sum(pd.isna(self.actual))
        self.axis = 1
        self.plot = plot
        if self.column_type in [np.dtype('O'), ]:
            self.expected_uniqs = expected[column_name].unique()
            self.actual_uniqs = actual[column_name].unique()
    
    def jac(self):
        """ Function calculates Jaccard metric for specified column """
        x = set(self.expected_uniqs)
        y = set(self.actual_uniqs)
        return len(x.intersection(y)) / len(x.union(y))

    def plots(self, nulls, expected_percents, actual_percents, breakpoints, intervals):
        """Plots actual and expected distributions of specified column."""
        points = [i for i in breakpoints] 
        plt.figure(figsize=(15,7))
        plt.bar(np.arange(len(intervals))-0.15, expected_percents, label='expected', alpha=0.7, width=.3)
        plt.bar(np.arange(len(intervals))+0.15, actual_percents, label='actual', alpha=0.7, width=.3)
        plt.legend(loc='best')
        if self.column_type not in [np.dtype('O')]:
            plt.xticks(range(len(intervals)), intervals, rotation=90)
        else: 
            plt.xticks(range(len(points)), points, rotation=90)
        plt.title(self.column_name)
        plt.show()

    def psi_num(self):
        """
        Calculate the PSI for a single variable
        Args:
           expected_array: numpy array of original values
           actual_array: numpy array of new values, same size as expected
           buckets: number of percentile ranges to bucket the values into
        Returns:
           psi_value: calculated PSI value
        """
        buckets = 10
        breakpoints = np.arange(0, (buckets)/10, 0.1)

        if self.expected_nulls == self.expected_len and self.actual_nulls != self.actual_len:
            breakpoints = np.array(list(sorted(set(np.nanquantile(self.actual, breakpoints)))))
        else:
            breakpoints = np.array(list(sorted(set(np.nanquantile(self.expected, breakpoints)))))

        actual_nulls = self.actual_nulls / self.actual_len
        expected_nulls = self.expected_nulls / self.expected_len
        breakpoints = np.concatenate(([-np.inf], breakpoints, [np.inf]))
        expected_percents = np.histogram(self.expected, breakpoints)
        actual_percents = np.histogram(self.actual, breakpoints)

        expected_percents = [p/self.expected_len for p in expected_percents[0]]
        actual_percents = [p/self.actual_len for p in actual_percents[0]]
        if self.expected_nulls==0 and actual_nulls==expected_nulls:
            expected_percents=expected_percents
            actual_percents=actual_percents
            nulls = False
        else:
            expected_percents.append(expected_nulls)
            actual_percents.append(actual_nulls)
            nulls = True
            
        points = [i for i in breakpoints]
        intervals = [f"({np.round(points[i], 5)};{np.round(points[i+1], 5)})" for i in range(len(points)-1)]
        if nulls:
                intervals = np.append(intervals, 'empty_values')

        if self.plot:
            self.plots(nulls, expected_percents, actual_percents, breakpoints, intervals)
            
        psi_value, psi_dict  = psi(expected_percents, actual_percents, breakpoints)
        new_cats = []
        abs_cats = []
        return psi_value, psi_dict, new_cats, abs_cats

    def uniq_psi(self):
        """ Counts psi for categorical  unique counts > 100 """
        actual_nulls = self.actual_nulls / self.actual_len
        expected_nulls = self.expected_nulls / self.expected_len
        actual_not_nulls_arr = self.actual[~pd.isna(self.actual)]
        expected_not_nulls_arr = self.expected[~pd.isna(self.expected)]
        actual_not_nulls = len(actual_not_nulls_arr) / self.actual_len
        expected_not_nulls = len(expected_not_nulls_arr) / self.expected_len
        expected_percents = [expected_not_nulls, expected_nulls]
        actual_percents = [actual_not_nulls, actual_nulls]
        breakpoints = ['good_data', 'nulls']
        if self.plot:
            self.plots(False, expected_percents, actual_percents, breakpoints, breakpoints)

        psi_value, psi_dict = psi(expected_percents, actual_percents, breakpoints)

        jac_metric = self.jac()
        new_cats, abs_cats = [], []
        
        if psi_value >= 0.2:
            psi_value = psi_value
            psi_dict.update({'metric': 'stability_index'})
        else:
            psi_value = 1 - jac_metric
            psi_dict.update({'metric': 'unique_index'})
        return psi_value, psi_dict, new_cats, abs_cats

    def psi_categ(self):
        """ Counts psi for categorical data exclude unique counts > 100 """
        expected_uniq_count = len(self.expected_uniqs)
        actual_uniq_count = len(self.actual_uniqs)

        if expected_uniq_count > 100 or actual_uniq_count > 100:
            psi_value, psi_dict, new_cats, abs_cats = self.uniq_psi()
            return psi_value, psi_dict, new_cats, abs_cats

        expected_dict = pd.DataFrame(self.expected, columns=[self.column_name]).groupby(self.column_name) \
            [self.column_name].count().sort_values(ascending=False).to_dict()
        actual_dict = pd.DataFrame(self.actual, columns=[self.column_name]).groupby(self.column_name) \
            [self.column_name].count().sort_values(ascending=False).to_dict()
        breakpoints = list(set(list(expected_dict.keys()) + list(actual_dict.keys())))
        new_cats = [k for k in actual_dict.keys() if k not in expected_dict.keys()]
        abs_cats = [k for k in expected_dict.keys() if k not in actual_dict.keys()]
        expected_dict_re = dict()
        actual_dict_re = dict()
        for b in breakpoints:
            if b in expected_dict and b not in actual_dict:
                expected_dict_re.update({b : expected_dict[b]})
                actual_dict_re.update({b: 0})
            elif b not in expected_dict and b in actual_dict:
                expected_dict_re.update({b : 0})
                actual_dict_re.update({b: actual_dict[b]})
            elif b in expected_dict and b in actual_dict:
                actual_dict_re.update({b: actual_dict[b]})
                expected_dict_re.update({b : expected_dict[b]})
        category_names = [c for c in expected_dict_re.keys()]
        groups = {}
        g_counts = len(category_names)
        group_num = 20
        if g_counts <= group_num:
            for g_n, val in enumerate(category_names):
                groups[val] = g_n
        else:
            group_size = np.floor(g_counts / group_num)
            current_pos = 0
            reminder = g_counts % group_num
            for g_n in range(group_num):
                if g_n < group_num - reminder:
                    group_values = category_names[int(current_pos): int(current_pos + group_size)]
                    current_pos += group_size
                else:
                    group_values = category_names[int(current_pos): int(current_pos + group_size + 1)]
                    current_pos += group_size + 1
                for val in group_values:
                    groups[val] = g_n
        group_sum = 0
        exp_dict = {}
        act_dict = {}
        group_re = -1
        cat_group_name = ''
        group_name_re = ''
        for  k, v in groups.items():
            current_group = v
            if current_group == group_re:
                group_re = v
                exp_dict.pop(group_name_re, None)
                act_dict.pop(group_name_re, None)
                cat_group_name = cat_group_name + ', ' + str(k)
                group_sum_exp += expected_dict_re[k]
                group_sum_act += actual_dict_re[k]
                exp_dict.update({cat_group_name:group_sum_exp})
                act_dict.update({cat_group_name:group_sum_act})
                group_name_re = cat_group_name
            else:
                group_name_re = str(k)
                group_re = v
                cat_group_name = str(k)
                group_sum_exp = expected_dict_re[k]
                group_sum_act = actual_dict_re[k]
                exp_dict.update({cat_group_name:group_sum_exp})
                act_dict.update({cat_group_name:group_sum_act})
        expected_percents = [e / self.expected_len for e in exp_dict.values()]
        actual_percents = [a / self.actual_len for a in act_dict.values()]

        breakpoints = [e for e in exp_dict.keys()]
        
        if self.plot:
            self.plots(False, expected_percents, actual_percents, breakpoints, breakpoints)

        psi_value, psi_dict  = psi(expected_percents, actual_percents, breakpoints)

        return psi_value, psi_dict, new_cats, abs_cats
    
    def psi_result(self):
        """ Functions return check results depends on column dtype."""
        if len(self.expected_shape) == 1:
            psi_values = np.empty(len(self.expected_shape))
        else:
            psi_values = np.empty(self.expected_shape[self.axis])

        for i in range(0, len(psi_values)):
            if self.column_type in [np.dtype('O')] or (self.expected_nulls == self.expected_len and
                                                       self.actual_nulls == self.actual_len):
                psi_values, psi_dict, new_cats, abs_cats = self.psi_categ()
            else:
                psi_values, psi_dict, new_cats, abs_cats = self.psi_num()
        return round(psi_values,2), psi_dict, new_cats, abs_cats
    