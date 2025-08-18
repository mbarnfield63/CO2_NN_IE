import pandas as pd
import glob
import sys
import time
import concurrent.futures


atomic_masses = {
    'C12': 12.0,
    'C13': 13.003355,
    'O16': 15.994915,
    'O17': 16.999132,
    'O18': 17.999161
}

isotopologue_atomic_masses = {
    626: [atomic_masses['O16'], atomic_masses['C12'], atomic_masses['O16']],
    627: [atomic_masses['O16'], atomic_masses['C12'], atomic_masses['O17']],
    628: [atomic_masses['O16'], atomic_masses['C12'], atomic_masses['O18']],
    636: [atomic_masses['O16'], atomic_masses['C13'], atomic_masses['O16']],
    637: [atomic_masses['O16'], atomic_masses['C13'], atomic_masses['O17']],
    638: [atomic_masses['O16'], atomic_masses['C13'], atomic_masses['O18']],
    727: [atomic_masses['O17'], atomic_masses['C12'], atomic_masses['O17']],
    728: [atomic_masses['O17'], atomic_masses['C12'], atomic_masses['O18']],
    737: [atomic_masses['O17'], atomic_masses['C13'], atomic_masses['O17']],
    738: [atomic_masses['O17'], atomic_masses['C13'], atomic_masses['O18']],
    828: [atomic_masses['O18'], atomic_masses['C12'], atomic_masses['O18']],
    838: [atomic_masses['O18'], atomic_masses['C13'], atomic_masses['O18']],
}


def mu1(mass_o_1, mass_c, mass_o_2):
    """
    Symmetric stretch, average of reduced atomic_masses for each side.
    """
    side1 = (mass_o_1 * mass_c) / (mass_o_1 + mass_c)
    side2 = (mass_o_2 * mass_c) / (mass_o_2 + mass_c)
    return (side1 + side2) / 2


def mu2(mass_o_1, mass_c, mass_o_2):
    """
    Bend, reduced atomic_masses for just oxygens.
    """
    return (mass_o_1 * mass_o_2) / (mass_o_1 + mass_o_2)


def mu3(mass_o_1, mass_c, mass_o_2):
    """
    Asymmetric stretch, combined mu for both sides.
    """
    return ((mass_o_1 + mass_o_2) * mass_c) / (mass_o_1 + mass_o_2 + mass_c)


def mu_all(mass_o_1, mass_c, mass_o_2):
    """
    Combined reduced mass for all three atoms.
    """
    return (mass_o_1 * mass_c * mass_o_2) / (mass_o_1 + mass_c + mass_o_2)


mu1_main = mu1(atomic_masses['O16'], atomic_masses['C12'], atomic_masses['O16'])
mu2_main = mu2(atomic_masses['O16'], atomic_masses['C12'], atomic_masses['O16'])
mu3_main = mu3(atomic_masses['O16'], atomic_masses['C12'], atomic_masses['O16'])
mu_all_main = mu_all(atomic_masses['O16'], atomic_masses['C12'], atomic_masses['O16'])


def prepare_atomic_features(isotopologue):
    # Add mass_c and mass_o columns to isotopologue
    isotopologue['mass_o_1'] = isotopologue['iso'].map(
        lambda x: isotopologue_atomic_masses[x][0])
    isotopologue['mass_c'] = isotopologue['iso'].map(
        lambda x: isotopologue_atomic_masses[x][1])
    isotopologue['mass_o_2'] = isotopologue['iso'].map(
        lambda x: isotopologue_atomic_masses[x][2])

    isotopologue['mu1'] = mu1(
        isotopologue['mass_o_1'], isotopologue['mass_c'], isotopologue['mass_o_2'])
    isotopologue['mu2'] = mu2(
        isotopologue['mass_o_1'], isotopologue['mass_c'], isotopologue['mass_o_2'])
    isotopologue['mu3'] = mu3(
        isotopologue['mass_o_1'], isotopologue['mass_c'], isotopologue['mass_o_2'])

    isotopologue['mu_all'] = mu_all(
        isotopologue['mass_o_1'], isotopologue['mass_c'], isotopologue['mass_o_2'])

    isotopologue['mu1_ratio'] = isotopologue['mu1'] / mu1_main
    isotopologue['mu2_ratio'] = isotopologue['mu2'] / mu2_main
    isotopologue['mu3_ratio'] = isotopologue['mu3'] / mu3_main

    isotopologue['mu_all_ratio'] = isotopologue['mu_all'] / mu_all_main

    # One-hot encoding for atomic_masses & symmetries
    mass_cols = ['mass_c', 'mass_o_1', 'mass_o_2']
    for mass_col in mass_cols:
        # Create one-hot encoding for each mass
        one_hot = pd.get_dummies(isotopologue[mass_col], prefix=mass_col)
        # Concatenate the one-hot encoding with the original DataFrame
        isotopologue = pd.concat([isotopologue, one_hot], axis=1)
        # Drop the original mass column
        isotopologue.drop(columns=[mass_col], inplace=True)

    # One-hot encoding for e/f
    if 'e_f' in isotopologue.columns:
        ef_one_hot = pd.get_dummies(
            isotopologue['e_f']).astype(bool)
        isotopologue = pd.concat([isotopologue, ef_one_hot], axis=1)
        isotopologue.drop(columns=['e_f'], inplace=True)
    
    # One-hot encoding for symmetries
    if 'tot_sym' in isotopologue.columns:
        symmetry_one_hot = pd.get_dummies(
            isotopologue['tot_sym'], prefix='Sym')
        isotopologue = pd.concat([isotopologue, symmetry_one_hot], axis=1)
        isotopologue.drop(columns=['tot_sym'], inplace=True)

    return isotopologue


def create_matching_key(df):
    """Create a matching key based on quantum numbers and e/f symmetry"""
    afgl_cols = [col for col in df.columns if col.startswith('AFGL')]
    
    # Create matching key
    key_cols = ['J'] + afgl_cols
    if 'e_f' in df.columns:
        key_cols.append('e_f')
    
    # Create a tuple of values for matching
    df['match_key'] = df[key_cols].apply(lambda row: tuple(row), axis=1)
    return df


if __name__ == "__main__":
    time_start = time.time()
    if len(sys.argv) != 2:
        print("Usage: python data_preprocessing.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    # Get all files in directory
    files = glob.glob(f"{directory}/*")
    
    # Separate dictionaries for ma and ca files by isotopologue
    ma_files = {}
    ca_files = {}

    def process_file(filename):
        if filename.endswith('ma.txt'):
            iso = int(filename.split('CO2_')[1].split('_')[0])
            df = pd.read_csv(filename, sep='\t')
            df['iso'] = iso
            return ('ma', iso, df)
        elif filename.endswith('ca.txt'):
            iso = int(filename.split('CO2_')[1].split('_')[0])
            df = pd.read_csv(filename, sep='\t')
            df['iso'] = iso
            return ('ca', iso, df)
        else:
            print(f"Skipping file: {filename}")
            return (None, None, None)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_file, files))

    # Organize files by isotopologue
    for kind, iso, df in results:
        if kind == 'ma' and df is not None:
            ma_files[iso] = df
        elif kind == 'ca' and df is not None:
            ca_files[iso] = df

    # Check if main isotopologue (626) exists
    if 626 not in ma_files or 626 not in ca_files:
        print("Error: Main isotopologue (626) ma or ca file not found!")
        sys.exit(1)

    # Prepare main isotopologue data for matching
    main_ma = ma_files[626].copy()
    main_ca = ca_files[626].copy()
    
    # Create matching keys
    main_ma = create_matching_key(main_ma)
    main_ca = create_matching_key(main_ca)
    
    # Create lookup dictionaries for main isotopologue energies
    main_ma_lookup = main_ma.set_index('match_key')['E'].to_dict()
    main_ca_lookup = main_ca.set_index('match_key')['E'].to_dict()

    # Process minor isotopologues
    final_dfs = []
    
    for iso, ma_df in ma_files.items():
        if iso == 626:  # Skip main isotopologue
            continue
            
        print(f"Processing isotopologue {iso}...")
        
        # Create matching key for minor isotopologue
        ma_df = create_matching_key(ma_df)
        
        # Add main isotopologue energy values
        ma_df['E_Ma_main'] = ma_df['match_key'].map(main_ma_lookup)
        ma_df['E_Ca_main'] = ma_df['match_key'].map(main_ca_lookup)
        
        # Drop rows where we couldn't find matches in main isotopologue
        initial_count = len(ma_df)
        ma_df = ma_df.dropna(subset=['E_Ma_main', 'E_Ca_main'])
        final_count = len(ma_df)
        
        if initial_count > final_count:
            print(f"  Dropped {initial_count - final_count} unmatched records for isotopologue {iso}")
        
        # Rename energy columns
        ma_df.rename(columns={'E': 'E_Ma_iso'}, inplace=True)
        
        # Drop the matching key as it's no longer needed
        ma_df.drop(columns=['match_key'], inplace=True)
        
        # Add to final list
        final_dfs.append(ma_df)
        print(f"  Added {len(ma_df)} records for isotopologue {iso}")

    # Concatenate all minor isotopologue dataframes
    if not final_dfs:
        print("No minor isotopologue data found!")
        sys.exit(1)
        
    combined_df = pd.concat(final_dfs, ignore_index=True)
    
    # Prepare atomic features
    combined_df = prepare_atomic_features(combined_df)

    # Columns to drop
    columns_to_drop = ['ID', 'unc', '??', 'Source']
    
    combined_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # Rename columns for consistency
    combined_df.rename(columns={'Sym_A"': 'Sym_Adp', "Sym_A'": 'Sym_Ap'}, inplace=True)

    # Save the processed dataframe
    combined_df.to_csv(f'Data/CO2_minor_isos_ma.txt', index=False)

    print(f"Dataset created with {len(combined_df)} total records from {len(final_dfs)} minor isotopologues")
    print(f"All files processed in {time.time() - time_start:.2f} seconds.")