import pandas as pd

MASSES = {
    'C12': 12.0,
    'C13': 13.003355,
    'O16': 15.994915,
    'O17': 16.999132,
    'O18': 17.999161
}

isotopologue_masses = {
    26: [MASSES['C12'], MASSES['O16']],
    27: [MASSES['C12'], MASSES['O17']],
    28: [MASSES['C12'], MASSES['O18']],
    36: [MASSES['C13'], MASSES['O16']],
    37: [MASSES['C13'], MASSES['O17']],
    38: [MASSES['C13'], MASSES['O18']],
}

mu_main = (masses := [MASSES['C12'], MASSES['O16']])[
    0] * masses[1] / sum(masses)


def prepare_data(df_all):

    # Extract data for isotopologue 26 from all_combined
    main = df_all[df_all['iso'] == 26].copy()
    main = main[['v', 'J', 'E_Ca', 'E_Ma']]
    # Remove 26 from all_combined
    minor_all = df_all[df_all['iso'] != 26].copy()
    minor_all = minor_all[['iso', 'v', 'J', 'E_Ca', 'E_Ma']]

    # Combine minor_all with main on v and J, adding E_duo_main and E_marvel_main columns
    main_renamed = main.rename(columns={
        'E_Ca': 'E_Ca_main',
        'E_Ma': 'E_Ma_main'
    })
    # Merge main_renamed with minor_all on v and J
    minor_renamed = minor_all.rename(columns={
        'E_Ca': 'E_Ca_iso',
        'E_Ma': 'E_Ma_iso'
    })
    minor_all = minor_renamed.merge(
        main_renamed[['v', 'J', 'E_Ca_main', 'E_Ma_main']],
        on=['v', 'J'],
        how='left'
    )

    # Drop any row with NaN values in any column
    minor_all.dropna(inplace=True)
    # Reset index
    minor_all.reset_index(drop=True, inplace=True)

    # Calculate E_IE and Error_IE
    minor_all['E_IE'] = minor_all['E_Ca_iso'] + minor_all['E_Ma_main'] - minor_all['E_Ca_main']
    minor_all['Error_IE'] = minor_all['E_Ma_iso'] - minor_all['E_IE']

    return minor_all, main


def prepare_atomic_features(minor_all):
    # Add mass_c and mass_o columns to minor_all
    minor_all['mass_c'] = minor_all['iso'].map(
        lambda x: isotopologue_masses[x][0])
    minor_all['mass_o_1'] = minor_all['iso'].map(
        lambda x: isotopologue_masses[x][1])

    minor_all['mu'] = (minor_all['mass_c'] * minor_all['mass_o_1']) / \
        (minor_all['mass_c'] + minor_all['mass_o_1'])
    minor_all['mu_ratio'] = minor_all['mu'] / mu_main

    # One-hot encoding for masses
    mass_cols = ['mass_c', 'mass_o_1']
    for mass_col in mass_cols:
        # Create one-hot encoding for each mass
        one_hot = pd.get_dummies(minor_all[mass_col], prefix=mass_col)
        # Concatenate the one-hot encoding with the original DataFrame
        minor_all = pd.concat([minor_all, one_hot], axis=1)
        # Drop the original mass column
        minor_all.drop(columns=[mass_col], inplace=True)

    return minor_all
