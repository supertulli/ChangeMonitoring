from typing import Literal

import pandas as pd


def period_freq_tables(target_df:pd.DataFrame, target_column:str, period: Literal['YE', 'ME', 'W', 'D'], date_column:str='visit_start_date') -> tuple[pd.DataFrame, pd.DataFrame]:
    if period not in ['YE', 'ME', 'W', 'D']:
        raise ValueError("period must be one of 'YE', 'ME', 'W', 'D'")
    
    period_target_count_df = target_df.groupby([pd.Grouper(key=date_column,freq=period),target_column])[target_column].count().reset_index(name='count')
    period_target_count_df['year'] = period_target_count_df[date_column].dt.year
    index = ['year']
    if period != 'YE':
        period_target_count_df['month'] = period_target_count_df[date_column].dt.month
        index.append('month')
        if period == 'W':
            period_target_count_df['week'] = period_target_count_df[date_column].dt.isocalendar().week
            index.append('week')
        elif period == 'D':
            period_target_count_df['day'] = period_target_count_df[date_column].dt.day
            index.append('day')
    index.append(date_column)
    period_target_count_df.sort_values(by=date_column)
    abs_freq_table = period_target_count_df.pivot(index=index, 
                                                columns=target_column,
                                                values='count').fillna(0).astype('int')
    rel_freq_table = abs_freq_table.div(abs_freq_table.sum(axis=1), axis= 0)
    rel_freq_table.reset_index(level=[date_column],inplace=True)
    abs_freq_table.reset_index(level=[date_column],inplace=True)
    return abs_freq_table, rel_freq_table
    