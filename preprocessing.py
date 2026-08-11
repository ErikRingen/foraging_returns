import numpy as np
import pandas as pd
from pathlib import Path

def preprocess_data(
    returns_file: str | Path,
    recall_file: str | Path,
    kcal_file: str | Path | None,
    group_file: str | Path,
    camp_members_file: str | Path,
    days_in_camp_file: str | Path,
    combine_returns_recall: bool = True,
    foraging_only: bool = True,
    include_recall: bool = True,
    exclude_resource_indices: list[int] | None = None,
    exclude_top_harvest_of: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Preprocesses raw foraging data files into analysis-ready DataFrames.

    Two input shapes are supported:

    - Raw data (``raw_data/``): rows in returns/recall carry an ``index``
      column linking to ``kcal.csv``'s per-resource ``kcal.g`` lookup.
      Pass ``kcal_file`` to enable the merge.
    - Public data (``public_data/``): each row has a precomputed ``kcal``
      column. Pass ``kcal_file=None``.

    Args:
        returns_file: Path to the returns data CSV.
        recall_file: Path to the recall data CSV.
        kcal_file: Path to the kcal data CSV, or ``None`` if returns/recall
            already carry precomputed ``kcal`` columns.
        group_file: Path to the group activity data CSV.
        camp_members_file: Path to the camp members demographic data CSV.
        days_in_camp_file: Path to the days in camp data CSV.
        foraging_only: If True (canonical), effort/time allocation counts only
            trips with ``activity.type == 'foraging'``. If False, every
            out-of-camp trip counts (all-outings sensitivity variant).
        include_recall: If False, drop the recall (field-consumption) stream so
            production reflects in-camp returns only.
        exclude_resource_indices: Resource ``index`` values to drop from both
            returns and recall before kcal computation (raw-data shape only).
        exclude_top_harvest_of: Drop the largest single harvest (one date x
            contributor set, summed across packages) of the given resource
            ``index`` values (raw-data shape only), e.g. the exceptionally
            large oil-palm harvest reported in the manuscript.

    Returns:
        A tuple containing:
        - df_foragers: DataFrame with forager demographic info (id, sex, age).
        - df_time_agg: DataFrame with time allocation per forager per date (id, date, total.minutes).
        - df_production: DataFrame with combined foraging/recall production (group_id, kcal, type, date).
        - df_days_long: DataFrame with daily camp presence records.
    """ 
    
    def _cols_to_lower(df):
        df.columns = [col.lower() for col in df.columns]
        return df

    # Read the data. `kcal_file` is optional: when both returns and recall
    # already carry a precomputed `kcal` column (the public-data shape),
    # the resource-to-kcal lookup is unused.
    df_returns = _cols_to_lower(pd.read_csv(returns_file))
    df_recall = _cols_to_lower(pd.read_csv(recall_file))
    if kcal_file is not None:
        df_kcal = _cols_to_lower(pd.read_csv(kcal_file))
    else:
        df_kcal = None
    df_group = _cols_to_lower(pd.read_csv(group_file))
    df_camp_members = _cols_to_lower(pd.read_csv(camp_members_file))
    df_days_in_camp = _cols_to_lower(pd.read_csv(days_in_camp_file))

    # Normalise IDs to a single canonical column name. The raw data uses
    # `id`; the public dataset uses `anon_id` (anonymised). Downstream
    # code consistently reads `id`, so rename here.
    for _df in (df_returns, df_recall, df_group, df_camp_members):
        if 'anon_id' in _df.columns and 'id' not in _df.columns:
            _df.rename(columns={'anon_id': 'id'}, inplace=True)

    # Format date columns. The raw CSVs use DD/MM/YYYY (and tests use
    # DD.MM.YY). Using ``format='mixed'`` with ``dayfirst=True`` covers
    # both without the UserWarning that bare ``dayfirst=True`` triggers
    # under recent pandas, and locks the day-month interpretation so
    # 06/07/2018 cannot be silently misread as 7-Jun.
    def _parse_dates(series):
        return pd.to_datetime(series, format='mixed', dayfirst=True)
    df_group['date'] = _parse_dates(df_group['date'])
    df_returns['date'] = _parse_dates(df_returns['date'])
    df_recall['date'] = _parse_dates(df_recall['date'])

    # --- Sensitivity-variant filters ---
    if not include_recall:
        df_recall = df_recall.iloc[0:0].copy()

    if exclude_resource_indices is not None or exclude_top_harvest_of is not None:
        if 'index' not in df_returns.columns:
            raise ValueError(
                "Resource exclusion requires the raw-data shape with a "
                "resource `index` column; the public dataset strips it."
            )
    if exclude_resource_indices is not None:
        excl = set(exclude_resource_indices)
        df_returns = df_returns[~df_returns['index'].isin(excl)].copy()
        if 'index' in df_recall.columns:
            df_recall = df_recall[~df_recall['index'].isin(excl)].copy()
    if exclude_top_harvest_of is not None:
        # "Harvest" is the model's returns observation: one date x contributor
        # set, which may bundle several food packages. Selecting the heaviest
        # single *package* would target a different (smaller) event than the
        # exceptionally large group harvest reported in the manuscript.
        target = df_returns['index'].isin(set(exclude_top_harvest_of))
        if target.any():
            keyed = df_returns.assign(
                _date_str=df_returns['date'].dt.strftime('%Y-%m-%d'),
                _pooled=np.round(
                    df_returns['pooled_group'].fillna(0), 0).astype(int),
            )
            keyed['_id'] = keyed['id'].astype(str)
            coop = keyed[keyed['_pooled'] != 0]
            ids_by_pkg = (
                coop.groupby(['_date_str', '_pooled'])['_id']
                .agg(lambda x: '_'.join(sorted(x.unique())))
                .rename('_ids')
                .reset_index()
            )
            keyed = keyed.merge(ids_by_pkg, on=['_date_str', '_pooled'],
                                how='left')
            keyed['_harvest'] = np.where(
                keyed['_pooled'] == 0,
                keyed['_date_str'] + '_' + keyed['_id'],
                keyed['_date_str'] + '_' + keyed['_ids'].astype(str),
            )
            # One row per package before summing (cooperative packages repeat
            # their weight on every contributor's row).
            pkgs = keyed[keyed['index'].isin(set(exclude_top_harvest_of))]
            pkgs = pd.concat([
                pkgs[pkgs['_pooled'] == 0],
                pkgs[pkgs['_pooled'] != 0].drop_duplicates(
                    ['_harvest', '_pooled', 'index', 'net_food_weight_gram']),
            ])
            mass = pkgs.groupby('_harvest')['net_food_weight_gram'].sum()
            top_harvest = mass.idxmax()
            drop = (keyed['_harvest'] == top_harvest) & target.values
            print(f"Processing: dropping largest harvest '{top_harvest}' "
                  f"({mass.max() / 1000:.1f} kg across "
                  f"{int(drop.sum())} contributor-rows).")
            df_returns = df_returns[~drop.values].copy()

    # --- Forager Demographics ---
    # Accept either `birthyear` (raw_data convention) or `age` (public_data
    # ships ages directly to avoid leaking exact birth year while still
    # enabling model refits).
    if 'age' not in df_camp_members.columns:
        df_camp_members['age'] = 2018 - df_camp_members['birthyear']
    df_foragers = df_camp_members[['id', 'sex', 'age']].copy()  # Use .copy() to avoid SettingWithCopyWarning
    df_foragers['sex'] = df_foragers['sex'].map({'M': 'male', 'F': 'female'})  # Use map for clarity
    df_foragers['id'] = df_foragers['id'].astype(str)

    # Age measurement uncertainty: ~1 year for ages < 20, ~5 years for ages >= 20
    # (based on estimation method: 1-year intervals for young, 10-year for older)
    df_foragers['age_sigma'] = np.where(
        df_foragers['age'] < 20,
        1.0,
        5.0,
    )

    # --- Days in Camp ---
    # First column is the participant ID, but its label varies across data
    # shapes: `id/date` (raw) or `anon_id` (public). Normalise it.
    _id_col = df_days_in_camp.columns[0]
    if _id_col != 'forager_id':
        df_days_in_camp.rename(columns={_id_col: 'forager_id'}, inplace=True)
    df_days_long = pd.melt(df_days_in_camp.drop(columns=['total']), 
                           id_vars=['forager_id'], 
                           var_name='date_str', 
                           value_name='in_camp')
    df_days_long['date'] = pd.to_datetime(df_days_long['date_str'], format='%d/%m/%Y')
    df_days_long['forager_id'] = df_days_long['forager_id'].astype(str)
    df_days_long.drop(columns=['date_str'], inplace=True)

    # --- Time Allocation ---
    # Prepare group data for time aggregation. Match activity.type case-
    # insensitively after stripping whitespace, so "foraging", "Foraging ",
    # etc. all count.
    activity_normalized = (
        df_group['activity.type'].astype(str).str.strip().str.casefold()
    )
    if foraging_only:
        df_group_time = df_group[activity_normalized == 'foraging'].copy()
    else:
        df_group_time = df_group.copy()
    df_group_time['id'] = df_group_time['id'].astype(str)

    # Create multi-index for all potential forager-date combinations
    all_forager_ids = df_group_time['id'].unique()
    all_dates = df_group_time['date'].unique()
    multi_index = pd.MultiIndex.from_product([all_forager_ids, all_dates], names=['id', 'date'])

    # Aggregate time, reindex to include all combinations, and sort
    df_time_agg = df_group_time.groupby(['id', 'date']).agg({'total.minutes': 'sum'})
    df_time_agg = df_time_agg.reindex(multi_index).reset_index().sort_values(by=['id', 'date'])

    # Merge with days_in_camp to identify time spent outside camp
    df_time_agg = pd.merge(
        df_time_agg,
        df_days_long[['forager_id', 'date', 'in_camp']],
        left_on=['id', 'date'],
        right_on=['forager_id', 'date'],
        how='left'
    )

    # Fill NaN time with 0 *only* if the forager was in camp (in_camp == 1)
    fill_condition = (df_time_agg['total.minutes'].isna()) & (df_time_agg['in_camp'] == 1)
    df_time_agg.loc[fill_condition, 'total.minutes'] = 0

    # Clean up temporary columns
    df_time_agg.drop(columns=['forager_id', 'in_camp'], inplace=True)

    # --- Production Data (Foraging Returns & Recall) ---
    # Two supported input shapes:
    #
    # (A) Raw data (raw_data/): rows carry an `index` column linking to
    #     `kcal.csv`'s per-resource `kcal.g` lookup, plus the original
    #     weight columns (`net_food_weight_gram`, `total_weight_grams`).
    #     We compute `kcal = kcal.g × weight` here.
    #
    # (B) Public data (public_data/): the resource lookup is removed, but
    #     each row has `kcal` precomputed during anonymisation. We simply
    #     use it.
    #
    # Auto-detect shape per stream and unify on a `kcal` column.
    for df_name, df, weight_col in [
        ("returns", df_returns, "net_food_weight_gram"),
        ("recall",  df_recall,  "total_weight_grams"),
    ]:
        if 'kcal' in df.columns:
            df['kcal'] = pd.to_numeric(df['kcal'], errors='coerce')
            n_bad = int(df['kcal'].isna().sum())
            if n_bad:
                print(f"Processing: {df_name}: {n_bad} rows with "
                      f"unparseable kcal (will be dropped from production sum).")
        else:
            if weight_col not in df.columns:
                raise KeyError(
                    f"{df_name} stream is missing both `kcal` and the "
                    f"weight column '{weight_col}'. Columns present: "
                    f"{list(df.columns)}"
                )
            df_kcal_local = df_kcal[['index', 'kcal.g']]
            df = df.merge(df_kcal_local, on='index', how='left')
            df['kcal.g'] = pd.to_numeric(df['kcal.g'], errors='coerce')
            n_bad = int(df['kcal.g'].isna().sum())
            if n_bad:
                print(f"Processing: {df_name}: {n_bad} rows with "
                      f"unparseable kcal.g (will be dropped from production sum).")
            df['kcal'] = df['kcal.g'] * df[weight_col]
            df.drop(columns=['kcal.g'], inplace=True, errors='ignore')
        df.drop(columns=[weight_col, 'index', 'gift'],
                inplace=True, errors='ignore')
        if df_name == "returns":
            df_foraging_merged = df
        else:
            df_recall_merged = df

    # --- Create group_id for foraging and recall ----------------------
    # Each food package's group_id encodes (date, contributing-forager-set).
    # For solo trips (`pooled_group == 0`) the format is "YYYY-MM-DD_<id>";
    # for cooperative trips it is "YYYY-MM-DD_<sorted_ids>". The same
    # construction is applied to both data streams.

    def _add_group_id(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['pooled_group'] = (
            np.round(df['pooled_group'].fillna(0), 0).astype(int).astype(str)
        )
        df['id'] = df['id'].astype(str)
        df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')

        non_solo = df[df['pooled_group'] != '0']
        if len(non_solo) > 0:
            agg = (
                non_solo
                .groupby(['date_str', 'pooled_group'])['id']
                .agg(lambda x: '_'.join(sorted(x.unique())))
                .reset_index()
                .rename(columns={'id': 'aggregated_ids'})
            )
            df = pd.merge(df, agg, on=['date_str', 'pooled_group'], how='left')
        else:
            df['aggregated_ids'] = pd.NA

        is_solo = df['pooled_group'] == '0'
        df['group_id'] = np.where(
            is_solo,
            df['date_str'] + '_' + df['id'],
            df['date_str'] + '_' + df['aggregated_ids'],
        )
        return df.drop(columns=['date_str', 'aggregated_ids'], errors='ignore')

    df_foraging_merged = _add_group_id(df_foraging_merged)
    df_recall_merged = _add_group_id(df_recall_merged)

    # Aggregate production per group_id.
    # The long table has one row per (forager, food package). For cooperative
    # packages, the package weight is repeated across every contributor — we
    # must collapse those forager replicates to one row before summing or we
    # under-count. Solo rows already carry one row per (forager, day, item),
    # so we sum directly. Within a group_id (date + sorted forager set),
    # multiple distinct food packages are then summed to give group-level
    # daily kcal.
    def _aggregate_per_group(df):
        solo_rows = df[df['pooled_group'] == '0']
        coop_rows = df[df['pooled_group'] != '0']
        coop_deduped = coop_rows.drop_duplicates(subset=['group_id', 'pooled_group'])
        return (
            pd.concat([solo_rows, coop_deduped], ignore_index=True)
            .groupby('group_id', as_index=False)['kcal']
            .sum()
        )

    df_foraging_agg = _aggregate_per_group(df_foraging_merged)
    df_recall_agg = _aggregate_per_group(df_recall_merged)

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
        # A stream can be globally absent (e.g. include_recall=False)
        for stream in ('foraging', 'recall'):
            if stream not in df_pivot.columns:
                df_pivot[stream] = 0.0
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

    return df_foragers, df_time_agg, df_production, df_days_long
        





    


    
