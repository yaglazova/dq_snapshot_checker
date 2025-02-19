# Data drift detection package - snapshot_check
## This package developed in order to check data drift in two versions: for PySpark and Pandas.
## There are several types of methods used:

1) Population Stability Index:
    -PSI > 0.2 - significant drift detected!
    -0.1 < PSI  < 0.2 - slightly drift detected
    -PSI <  0.1 - no drift
2) Inex based on Jaccard metric:
    - close to zero - no drift
    - close to one - drift detected
3)  Atribution check for specified columns (only PySpark version)
    Performs join on unique value column or columns  and commpare their attributes for list of columns and show their similarity, where 1 means that attributes are equal.
