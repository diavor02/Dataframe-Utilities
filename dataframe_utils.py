from types import MethodType
from typing import List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.model_selection import KFold, StratifiedKFold


def summary(self: pd.DataFrame, categorical_columns: List[str], numerical_columns: List[str]) -> None:
    """Generate summary statistics and visualizations for DataFrame columns.
    
    For categorical columns: shows value counts and bar plots
    For numerical columns: shows box plots and histograms
    
    Args:
        categorical_columns: List of categorical column names
        numerical_columns: List of numerical column names
    """
    for col in categorical_columns:
        unique_counts = self[col].value_counts()
        
        print(f"Number of unique values in [{col}]: {len(unique_counts)}")
        print(f"Unique values in [{col}]:")
        print(unique_counts, end="\n\n")
        
        plt.figure(figsize=(7, 5))
        plt.bar(unique_counts.index, unique_counts.values)
        plt.title(f"Counts of unique values in {col}")
        plt.show()

    for col in numerical_columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        sns.boxplot(x=self[col], ax=ax1)
        ax1.set_title(f"Box plot for {col}")

        ax2.hist(self[col], bins='auto')
        ax2.set_title(f"Histogram for {col}")

        plt.tight_layout()
        plt.show()


def null_values(self: pd.DataFrame) -> None:
    """Visualize missing values in DataFrame using heatmap."""
    plt.figure(figsize=(15, 10))
    sns.heatmap(self.isna(), cmap='coolwarm')
    plt.title("Missing Values Heatmap")
    plt.show()


def sin_cos_transformation(self: pd.DataFrame, date_columns: List[str], periods: int) -> None:
    """Add sinusoidal transformations for date columns.
    
    Args:
        date_columns: List of date column names to transform
        periods: Number of periods for sinusoidal transformation
    """
    for col in date_columns:
        date_series = pd.to_datetime(self[col])
        day_of_year = date_series.dt.dayofyear
        angle = 2 * np.pi * day_of_year / periods
        
        self[f'{col}_day_sin'] = np.sin(angle)
        self[f'{col}_day_cos'] = np.cos(angle)
        
        self.drop(columns=col, inplace=True)


def correlation_matrix(self: pd.DataFrame, numerical_columns: List[str]) -> None:
    """Display correlation matrix heatmap for numerical columns."""
    corr_matrix = self[numerical_columns].corr()

    plt.figure(figsize=(15, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        fmt=".2f"
    )
    plt.title("Correlation Matrix")
    plt.show()


def determine_aggregate_parameters(
    std: bool,
    skewness: bool,
    kurt: bool,
    train_fold: pd.DataFrame,
    col: str,
    target_col: str
) -> pd.DataFrame:
    """Calculate aggregated statistics for target encoding.
    
    Args:
        std: Whether to calculate standard deviation
        skewness: Whether to calculate skewness
        kurt: Whether to calculate kurtosis
        train_fold: Training fold DataFrame
        col: Column to group by
        target_col: Target column name
        
    Returns:
        Aggregated DataFrame with calculated statistics
    """
    agg_funcs = ['mean', 'count']
    col_names = ['mean', 'count']

    if std:
        agg_funcs.append('std')
        col_names.append('std')
    if skewness:
        agg_funcs.append(lambda x: skew(x))
        col_names.append('skewness')
    if kurt:
        agg_funcs.append(lambda x: kurtosis(x))
        col_names.append('kurtosis')

    aggregates = train_fold.groupby(col)[target_col].agg(agg_funcs)
    aggregates.columns = col_names
    return aggregates


def target_enc_categorical_target(
    df: pd.DataFrame,
    col: str,
    target_col: str,
    smoothing: int,
    n_splits: int,
    random_state: int
) -> pd.DataFrame:
    """Target encoding for categorical targets using stratified K-Fold.
    
    Args:
        df: Input DataFrame
        col: Column to encode
        target_col: Target column name
        smoothing: Smoothing factor for encoding
        n_splits: Number of cross-validation folds
        random_state: Random seed for reproducibility
    """
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )
    
    for train_idx, valid_idx in skf.split(df, df[target_col]):
        train_fold = df.iloc[train_idx]
        global_mean = train_fold[target_col].mean()
        
        aggregates = train_fold.groupby(col)[target_col].agg(['mean', 'count'])
        smoothing_factor = (
            (aggregates['count'] * aggregates['mean'] + smoothing * global_mean) 
            / (aggregates['count'] + smoothing)
        )
        
        df.loc[valid_idx, f'{col}_enc'] = df.loc[valid_idx, col].map(smoothing_factor)
        
    return df


def target_enc_continuous_target(
    df: pd.DataFrame,
    col: str,
    target_col: str,
    smoothing: int,
    std: bool,
    skewness: bool,
    kurtosis_flag: bool,
    n_splits: int,
    random_state: int
) -> pd.DataFrame:
    """Target encoding for continuous targets with optional statistics.
    
    Args:
        df: Input DataFrame
        col: Column to encode
        target_col: Target column name
        smoothing: Smoothing factor for encoding
        std: Whether to include standard deviation
        skewness: Whether to include skewness
        kurtosis_flag: Whether to include kurtosis
        n_splits: Number of cross-validation folds
        random_state: Random seed for reproducibility
    """
    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )
    
    for train_idx, valid_idx in kf.split(df):
        train_fold = df.iloc[train_idx]
        global_mean = train_fold[target_col].mean()
        
        aggregates = determine_aggregate_parameters(
            std, skewness, kurtosis_flag, train_fold, col, target_col)
        
        smoothing_factor = (
            (aggregates['count'] * aggregates['mean'] + smoothing * global_mean)
            / (aggregates['count'] + smoothing))
        
        df.loc[valid_idx, f'{col}_enc'] = df.loc[valid_idx, col].map(smoothing_factor)
        
        if std:
            df.loc[valid_idx, f'{col}_enc_std'] = df.loc[valid_idx, col].map(aggregates['std'])
        if skewness:
            df.loc[valid_idx, f'{col}_enc_skew'] = df.loc[valid_idx, col].map(aggregates['skewness'])
        if kurtosis_flag:
            df.loc[valid_idx, f'{col}_enc_kurt'] = df.loc[valid_idx, col].map(aggregates['kurtosis'])
    
    return df


def target_encoding(
    self: pd.DataFrame,
    columns_to_encode: List[str],
    target_col: str,
    categorical_target: bool = False,
    std: bool = True,
    skewness: bool = True,
    kurtosis: bool = True,
    n_splits: int = 5,
    smoothing: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """Main target encoding function with configurable options.
    
    Args:
        columns_to_encode: Columns to apply target encoding to
        target_col: Target column name
        categorical_target: Whether target is categorical
        std: Whether to include standard deviation
        skewness: Whether to include skewness
        kurtosis: Whether to include kurtosis
        n_splits: Number of cross-validation folds
        smoothing: Smoothing factor for encoding
        random_state: Random seed for reproducibility
    """
    global_mean = self[target_col].mean()
    
    for col in columns_to_encode:
        if categorical_target:
            self = target_enc_categorical_target(
                self, col, target_col, smoothing, n_splits, random_state)
        else:
            self = target_enc_continuous_target(
                self, col, target_col, smoothing, std, 
                skewness, kurtosis, n_splits, random_state)

        self[col].fillna(global_mean, inplace=True)
        self.drop(columns=col, inplace=True)
    
