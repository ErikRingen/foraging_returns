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
    combine_returns_recall: bool = True,
    output_dir: str | Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Preprocesses raw foraging data files into analysis-ready DataFrames.

    Args:
        returns_file: Path to the returns data CSV.
        recall_file: Path to the recall data CSV.
        kcal_file: Path to the kcal data CSV.
        group_file: Path to the group activity data CSV.
        camp_members_file: Path to the camp members demographic data CSV.
        days_in_camp_file: Path to the days in camp data CSV.
        output_dir: Optional path to save the processed DataFrames.

    Returns:
        A tuple containing:
        - df_foragers: DataFrame with forager demographic info (id, sex, age).
        - df_time_agg: DataFrame with time allocation per forager per date (id, date, total.minutes).
        - df_production: DataFrame with combined foraging/recall production (group_id, kcal, type, date).
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

    # Format date columns
    df_group['date'] = pd.to_datetime(df_group['date'], dayfirst=True)
    df_returns['date'] = pd.to_datetime(df_returns['date'], dayfirst=True)
    df_recall['date'] = pd.to_datetime(df_recall['date'], dayfirst=True)

    # --- Forager Demographics ---
    df_camp_members['age'] = 2018 - df_camp_members['birthyear']
    df_foragers = df_camp_members[['id', 'sex', 'age']].copy() # Use .copy() to avoid SettingWithCopyWarning
    df_foragers['sex'] = df_foragers['sex'].map({'M': 'male', 'F': 'female'}) # Use map for clarity
    df_foragers['id'] = df_foragers['id'].astype(str)

    # --- Days in Camp --- 
    df_days_in_camp.rename(columns={'id/date': 'forager_id'}, inplace=True)
    df_days_long = pd.melt(df_days_in_camp.drop(columns=['total']), 
                           id_vars=['forager_id'], 
                           var_name='date_str', 
                           value_name='in_camp')
    df_days_long['date'] = pd.to_datetime(df_days_long['date_str'], format='%d/%m/%Y')
    df_days_long['forager_id'] = df_days_long['forager_id'].astype(str)
    df_days_long.drop(columns=['date_str'], inplace=True)

    # --- Time Allocation & Trip Counts --- 
    # Prepare group data for time aggregation
    df_group_time = df_group[df_group['activity.type'] == "Foraging"].copy()
    df_group_time['id'] = df_group_time['id'].astype(str)

    # Create multi-index for all potential forager-date combinations
    all_forager_ids = df_group_time['id'].unique()
    all_dates = df_group_time['date'].unique()
    multi_index = pd.MultiIndex.from_product([all_forager_ids, all_dates], names=['id', 'date'])

    # Aggregate time and trip counts per (forager, date)
    # Each row in raw groups.csv is one trip, so count rows = number of trips
    df_time_agg = df_group_time.groupby(['id', 'date']).agg(
        **{'total.minutes': ('total.minutes', 'sum'),
           'trip_count': ('total.minutes', 'size')}
    )
    df_time_agg = df_time_agg.reindex(multi_index).reset_index().sort_values(by=['id', 'date'])

    # Merge with days_in_camp to identify time spent outside camp
    df_time_agg = pd.merge(
        df_time_agg,
        df_days_long[['forager_id', 'date', 'in_camp']],
        left_on=['id', 'date'],
        right_on=['forager_id', 'date'],
        how='left'
    )

    # Fill NaN time/trips with 0 *only* if the forager was in camp (in_camp == 1)
    fill_condition = (df_time_agg['total.minutes'].isna()) & (df_time_agg['in_camp'] == 1)
    df_time_agg.loc[fill_condition, 'total.minutes'] = 0
    df_time_agg.loc[fill_condition, 'trip_count'] = 0

    # Ensure trip_count is integer (NaN -> keep as NaN for out-of-camp days)
    in_camp_mask = df_time_agg['trip_count'].notna()
    df_time_agg.loc[in_camp_mask, 'trip_count'] = df_time_agg.loc[in_camp_mask, 'trip_count'].astype(int)

    # Clean up temporary columns
    df_time_agg.drop(columns=['forager_id', 'in_camp'], inplace=True)

    # --- Production Data (Foraging Returns & Recall) ---
    # Merge returns/recall with kcal data
    df_foraging_merged = pd.merge(df_returns, df_kcal[['index', 'kcal.g']], on='index', how='left')
    df_recall_merged = pd.merge(df_recall, df_kcal[['index', 'kcal.g']], on='index', how='left')

    # Calculate total kcal
    for df in [df_foraging_merged, df_recall_merged]:
        df['kcal.g'] = pd.to_numeric(df['kcal.g'], errors='coerce')
        weight_col = 'net_food_weight_gram' if 'net_food_weight_gram' in df.columns else 'x1_unit_weight_grams'
        df['kcal'] = df['kcal.g'] * df[weight_col]
        # Clean up unnecessary columns
        df.drop(columns=['kcal.g', weight_col, 'index', 'gift'], inplace=True, errors='ignore') 

    # --- Create group_id for df_foraging_merged ---
    df = df_foraging_merged # Work with a clear reference
    df['pooled_group'] = np.round(df['pooled_group'].fillna(0), 0).astype(int).astype(str)
    df['id'] = df['id'].astype(str)
    df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
    non_solo_df = df[df['pooled_group'] != '0'].copy()
    if not non_solo_df.empty:
        group_forager_ids = non_solo_df.groupby(['date_str', 'pooled_group'])['id'] \
                                        .agg(lambda x: '_'.join(sorted(x.unique()))) \
                                        .reset_index() \
                                        .rename(columns={'id': 'aggregated_ids'})
        df = pd.merge(df, group_forager_ids, on=['date_str', 'pooled_group'], how='left')
    else:
        df['aggregated_ids'] = pd.NA 
    is_solo = df['pooled_group'] == '0'
    id_if_solo = df['date_str'] + '_' + df['id']
    id_if_group = df['date_str'] + '_' + df['aggregated_ids']
    df['group_id'] = np.where(is_solo, id_if_solo, id_if_group)
    df.drop(columns=['date_str', 'aggregated_ids'], inplace=True, errors='ignore')
    df_foraging_merged = df # Assign back the modified dataframe

    # --- Create group_id for df_recall_merged ---
    df = df_recall_merged # Work with a clear reference
    df['pooled_group'] = np.round(df['pooled_group'].fillna(0), 0).astype(int).astype(str)
    df['id'] = df['id'].astype(str)
    df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
    non_solo_df = df[df['pooled_group'] != '0'].copy()
    if not non_solo_df.empty:
        group_forager_ids = non_solo_df.groupby(['date_str', 'pooled_group'])['id'] \
                                        .agg(lambda x: '_'.join(sorted(x.unique()))) \
                                        .reset_index() \
                                        .rename(columns={'id': 'aggregated_ids'})
        df = pd.merge(df, group_forager_ids, on=['date_str', 'pooled_group'], how='left')
    else:
        df['aggregated_ids'] = pd.NA
    is_solo = df['pooled_group'] == '0'
    id_if_solo = df['date_str'] + '_' + df['id']
    id_if_group = df['date_str'] + '_' + df['aggregated_ids']
    df['group_id'] = np.where(is_solo, id_if_solo, id_if_group)
    df.drop(columns=['date_str', 'aggregated_ids'], inplace=True, errors='ignore')
    df_recall_merged = df # Assign back the modified dataframe

    # Aggregate production per group_id (taking mean kcal)
    df_foraging_agg = df_foraging_merged.groupby('group_id', as_index=False)['kcal'].mean()
    df_recall_agg = df_recall_merged.groupby('group_id', as_index=False)['kcal'].mean()

    # Combine foraging and recall production
    df_production = pd.concat([
        df_foraging_agg.assign(type='foraging'),
        df_recall_agg.assign(type='recall')
    ], ignore_index=True)

    if combine_returns_recall:
        # Create a pivot table to get foraging and recall values side by side
        df_pivot = df_production.pivot(index='group_id', columns='type', values='kcal').reset_index()
        # Fill missing values with 0 for groups that only have one type
        df_pivot = df_pivot.fillna(0)
        # Calculate total and foraging proportion
        df_pivot['total_kcal'] = df_pivot['foraging'] + df_pivot['recall']
        df_pivot['foraging_proportion'] = df_pivot['foraging'] / df_pivot['total_kcal']
        # Create final production dataframe with combined values
        df_production = df_pivot[['group_id', 'total_kcal', 'foraging_proportion']].rename(columns={'total_kcal': 'kcal'})
        df_production['type'] = 'combined'

    # --- Check Time vs. Returns & Append Zero Rows ---
    print("--- Checking Time vs Production & Appending Zeros ---")
    foraging_time_col = 'total.minutes'

    # 1. Get (id, date) pairs with foraging time > 0 from df_time_agg
    valid_time_entries = df_time_agg[
        (df_time_agg[foraging_time_col] > 0) & 
        (df_time_agg[foraging_time_col].notna())
    ].copy() # Use .copy() for safety
    valid_time_entries['id'] = valid_time_entries['id'].astype(str)
    valid_time_entries['date'] = pd.to_datetime(valid_time_entries['date'])
    time_agg_pairs = set(zip(valid_time_entries['id'], valid_time_entries['date']))

    # 2. Get (individual_id, date) pairs from df_production
    production_pairs = set()
    # Iterate through unique group_ids to avoid redundant parsing
    for group_id in df_production['group_id'].unique():
        parts = group_id.split('_', 1) 
        if len(parts) == 2:
            date_str, ids_str = parts
            try:
                date_obj = pd.to_datetime(date_str) 
                individual_ids = ids_str.split('_')
                for ind_id in individual_ids:
                    production_pairs.add((str(ind_id), date_obj))
            except ValueError:
                print(f"Warning: Could not parse date or IDs from group_id: {group_id}")
        else:
             print(f"Warning: Unexpected group_id format: {group_id}")

    # 3. Find (id, date) pairs with time logged but no production record
    time_without_production_pairs = time_agg_pairs - production_pairs

    # 4. Count and report
    num_time_without_production = len(time_without_production_pairs)
    print(f"Processing: Found {num_time_without_production} (forager, date) instances with time but no production.")

    # 5. Append zero-production rows if needed
    if num_time_without_production > 0:
        zero_production_rows = []
        for forager_id, forage_date in time_without_production_pairs:
            date_str = forage_date.strftime('%Y-%m-%d') 
            group_id = f"{date_str}_{forager_id}" # Implicitly solo trip
            new_row = {
                'group_id': group_id,
                'kcal': 0,
                'type': 'zero_inferred' # New type for these rows
            }
            zero_production_rows.append(new_row)

        df_zero_production = pd.DataFrame(zero_production_rows)
        df_production = pd.concat([df_production, df_zero_production], ignore_index=True)
        print(f"Processing: Appended {len(df_zero_production)} zero-inferred production rows.")
    print("--- End Time vs Production Check ---")

    # --- Final Filtering and Cleanup ---
    # Filter foragers to include only those present in time allocation data
    present_forager_ids = df_time_agg['id'].unique()
    df_foragers = df_foragers[df_foragers['id'].isin(present_forager_ids)].reset_index(drop=True)

    # Add date column to df_production (parsed from group_id)
    # Use error coercion for safety, although warnings handle most issues
    df_production['date'] = pd.to_datetime(df_production['group_id'].str.split('_').str[0], errors='coerce')
    # Report if any dates could not be parsed
    if df_production['date'].isna().any():
        print("Warning: Some dates could not be parsed from group_id in the final production df.")

    # add number of foragers in each group
    df_production['group_size'] = df_production['group_id'].str.split('_').apply(len) - 1

    # get set of foragers in each group
    df_production['forager_ids'] = df_production['group_id'].str.split('_').str[1:].apply(lambda x: set(x))

    # Remove any production rows where no forager ids appear in df_foragers
    valid_foragers_set = set(df_foragers['id'])
    df_production = df_production[df_production['forager_ids'].apply(lambda x: len(x.intersection(valid_foragers_set)) > 0)]

    # Remove foragers from df_days_long that are not in df_foragers
    df_days_long = df_days_long[df_days_long['forager_id'].isin(df_foragers['id'])]

    # --- Save Output (Optional) ---
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        df_foragers.to_csv(output_path / 'foragers.csv', index=False)
        df_time_agg.to_csv(output_path / 'time_allocation.csv', index=False)
        df_production.to_csv(output_path / 'production.csv', index=False)
        print(f"Processed files saved to: {output_path}")

    return df_foragers, df_time_agg, df_production, df_days_long
        





    


    
