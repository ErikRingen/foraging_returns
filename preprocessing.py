import numpy as np
import pandas as pd
from pathlib import Path

def preprocess_data(
    returns_file: str | Path,
    recall_file: str | Path,
    kcal_file: str | Path,
    group_file: str | Path,
    camp_members_file: str | Path,
    days_in_camp_file: str | Path,
    output_dir: str | Path = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Preprocess the data
    """ 
    
    def _cols_to_lower(df):
        df.columns = [col.lower() for col in df.columns]
        return df

    # Read the data
    df_returns = _cols_to_lower(pd.read_csv(returns_file))
    df_recall = _cols_to_lower(pd.read_csv(recall_file))
    df_kcal = _cols_to_lower(pd.read_csv(kcal_file))
    df_group = _cols_to_lower(pd.read_csv(group_file))
    df_camp_members = _cols_to_lower(pd.read_csv(camp_members_file))
    df_days_in_camp = _cols_to_lower(pd.read_csv(days_in_camp_file))
    # make dates have some format
    df_group['date'] = pd.to_datetime(df_group['date'], dayfirst=True)
    df_returns['date'] = pd.to_datetime(df_returns['date'], dayfirst=True)

    # Get forager demographic info ------------------------------------------------
    df_camp_members['age'] = df_camp_members['birthyear'].apply(lambda x: 2018 - x)
    df_foragers = df_camp_members[['id', 'sex', 'age']]
    df_foragers.head()

    # recode sex as "male" and "female"
    df_foragers['sex'] = df_foragers['sex'].apply(lambda x: "male" if x == "M" else "female")
    df_foragers.head()

    #unique_forager_ids = df_group_agg['id'].apply(lambda x: x.split('_')).explode().unique()

    df_foragers['id'] = df_foragers['id'].astype(str)
        #df_foragers = df_foragers[df_foragers['id'].isin(unique_forager_ids)]

    # Days in camp ---------------------------------------------------------------
    df_days_in_camp.rename(columns={'id/date': 'forager_id'}, inplace=True)
    # Melt the dataframe to long format
    df_days_long = pd.melt(df_days_in_camp.drop(columns=['total']), id_vars=['forager_id'], var_name='date_str', value_name='in_camp')
    # Convert date string to datetime objects, ensuring correct format
    df_days_long['date'] = pd.to_datetime(df_days_long['date_str'], format='%d/%m/%Y')
    df_days_long.drop(columns=['date_str'], inplace=True)
    # Ensure forager_id is string type to match df_time_agg['id']
    df_days_long['forager_id'] = df_days_long['forager_id'].astype(str)


    # Create a time allocation dataframe with dims (forager, date) --------------------------------------------
    #df_group['group'] = df_group['group'].astype(str)
    df_group['id'] = df_group['id'].astype(str)

    #solo_foraging = df_group['group'] == '0'
    #id_if_true = df_group['date'].astype(str) + '_' + df_group['id']
    #id_if_false = df_group['date'].astype(str) + '_' + df_group['group'] # Assuming group is integer/float, convert to str

    #df_group['group_id'] = np.where(solo_foraging, id_if_true, id_if_false)

    df_group_foraging = df_group[df_group['activity.type'] == "Foraging"]

    # Create all combinations of id and date
    ids = df_group_foraging['id'].unique()
    dates = df_group_foraging['date'].unique()
    multi_index = pd.MultiIndex.from_product([ids, dates], names=['id', 'date'])

    # Aggregate and reindex to include all combinations
    df_time_agg = df_group_foraging.groupby(['id', 'date']).agg({'total.minutes': 'sum'})
    df_time_agg = df_time_agg.reindex(multi_index).reset_index().sort_values(by=['date'])

    # Merge with days_in_camp data to fill NaNs conditionally
    df_time_agg = pd.merge(
        df_time_agg,
        df_days_long[['forager_id', 'date', 'in_camp']],
        left_on=['id', 'date'],
        right_on=['forager_id', 'date'],
        how='left'
    )

    # Fill NaN in 'total.minutes' with 0 only if the forager was in camp (in_camp == 1)
    fill_condition = (df_time_agg['total.minutes'].isna()) & (df_time_agg['in_camp'] == 1)
    df_time_agg.loc[fill_condition, 'total.minutes'] = 0

    # Drop the helper columns used for merging and condition
    df_time_agg.drop(columns=['forager_id', 'in_camp'], inplace=True)

    # get the number of unique ids within each group, split the ids into a list and count the number of unique ids
    #df_group_agg['id_count'] = df_group_agg['id'].apply(lambda x: len(x.split('_')))

    # Merge foraging and recall with kcal ------------------------------------------
    df_recall_merged = pd.merge(df_recall, df_kcal[['index', 'kcal.g']], on='index', how='left')
    df_foraging_merged = pd.merge(df_returns, df_kcal[['index', 'kcal.g']], on='index', how='left')

    df_merged['kcal.g'] = pd.to_numeric(df_merged['kcal.g'], errors='coerce')
    df_merged['kcal'] = df_merged['kcal.g'] * df_merged['net_food_weight_gram']
    df_merged = df_merged[df_merged["gift"] == 0].drop(columns=["gift"])

    df_merged['pooled_group'] = np.round(df_merged['pooled_group'].fillna(0), 0).astype(int).astype(str)
    df_merged['id'] = df_merged['id'].astype(str)

    solo_foraging = df_merged['pooled_group'] == '0'
    id_if_true = df_merged['date'].astype(str) + '_' + df_merged['id']
    id_if_false = df_merged['date'].astype(str) + '_' + df_merged['pooled_group'] # Assuming group is integer/float, convert to str

    df_merged['group_id'] = np.where(solo_foraging, id_if_true, id_if_false)

    # get the mean, because returns are repeated for each member of a group
    df_merged_agg = df_merged.groupby('group_id').agg({'kcal': 'mean'}).reset_index()

    # Join the two dataframes on the group_id column, fill missing values with 0
    df_joined = pd.merge(df_group_agg[['group_id', 'total.minutes']], df_merged_agg, on='group_id', how='left')

    df_joined["kcal"] = df_joined["kcal"].fillna(0)
    np.mean(df_joined['kcal'] == 0)

    ### Merge with camp_members

    # Get date from group_id
    df_group_agg["date"] = pd.to_datetime(df_group_agg["group_id"].str.split("_").str[0])
    df_joined["date"] = pd.to_datetime(df_joined["group_id"].str.split("_").str[0])

    # Rename cols


    if output_dir is not None:
        df_foragers.to_csv(output_dir + '/foragers.csv', index=False)
        df_group_agg.to_csv(output_dir + '/groups.csv', index=False)
        df_joined.to_csv(output_dir + '/returns.csv', index=False)

    return df_foragers, df_group_agg, df_joined
        





    


    
