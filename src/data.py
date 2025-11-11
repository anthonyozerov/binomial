import os
import pandas as pd
import numpy as np
from pdg_data import load_pdg_data


def dfs_from_folder(folder):
    path_list = os.listdir(folder)
    files = [path for path in path_list if path.endswith(".csv")]
    files.sort()
    dfs = [pd.read_csv(f"{folder}/{file}", comment="#") for file in files]
    return dfs, files


def load_noground_datasets():
    """
    Load all datasets and return dictionaries organizing them.
    
    Returns:
        tuple: (datasets, quantities, nice_names, nice_names_break)
            - datasets: dict mapping dataset names to lists of dataframes
            - quantities: dict mapping dataset names to lists of quantity names/files
            - nice_names: dict mapping dataset names to display names
            - nice_names_break: dict mapping dataset names to display names with line breaks
    """
    # Load PDG 2025 data
    pdg2025_stat_dfs, pdg2025_both_dfs, pdg2025_stat_quantities, pdg2025_both_quantities = load_pdg_data()
    
    # Initialize dictionaries
    datasets = {
        "pdg2025-stat": pdg2025_stat_dfs,
        "pdg2025-both": pdg2025_both_dfs,
    }
    quantities = {
        "pdg2025-stat": pdg2025_stat_quantities,
        "pdg2025-both": pdg2025_both_quantities,
    }
    
    # Mapping of dataset names to folder paths
    folder_mapping = {
        "baker-pdg2011-stat": "data/baker-pdg2011-stat",
        "baker-pdg2011-both": "data/baker-pdg2011-both",
        "bailey-pdg": "data/bailey/CsvData/Particle_Data",
        "bailey-pdg-stable": "data/bailey/CsvData/Particle_Data_Stable",
        "pdg1970": "data/pdg1970",
        "bipm-radionuclide": "data/bipm-radionuclide",
        "bailey-nuclear": "data/bailey/CsvData/Nuclear_Data",
        "bailey-interlab": "data/bailey/CsvData/Interlab_Data",
        "bailey-interlab-key": "data/bailey/CsvData/Interlab_Data_Key",
        "manylabs2": "data/psymetadata",
        "baker-medical": "data/baker-medical/clean",
        "bailey-medical": "data/bailey/CsvData/Medical_Data",
        "bailey-constants": "data/bailey/CsvData/Constants_Data",
        # "cochrane-dich": "data/cochrane/dich",
    }
    
    # Load datasets directly into dictionaries
    for name, folder in folder_mapping.items():
        dfs, files = dfs_from_folder(folder)
        datasets[name] = dfs
        quantities[name] = files
    
    # Special processing for baker-medical: add uncertainty column
    for df in datasets["baker-medical"]:
        df["uncertainty"] = np.sqrt(df["sigma2"])
    nice_names_break = {
        "pdg1970": "PDG 1970",
        "pdg2025-stat": "PDG 2025 (stat)",
        "pdg2025-both": "PDG 2025 (stat + syst)",
        "baker-pdg2011-stat": "Baker 2013\nPDG 2011 (stat)",
        "baker-pdg2011-both": "Baker 2013\nPDG 2011 (stat + syst)",
        "bailey-pdg": "Bailey 2017\nPDG",
        "bailey-pdg-stable": "Bailey 2017\nPDG (stable)",
        "bipm-radionuclide": "BIPM Radionuclide",
        "bailey-nuclear": "Bailey 2017\nnuclear",
        "bailey-interlab": "Bailey 2017\ninterlab",
        "bailey-interlab-key": "Bailey 2017\ninterlab (key)",
        "manylabs2": "Many Labs 2",
        "baker-medical": "Baker 2013\nmedical",
        "bailey-medical": "Bailey 2017\nmedical",
        "cochrane-dich": "CDSR: dichotomous",
        "bailey-constants": "Bailey 2017\nconstants",
    }
    # Generate nice_names from nice_names_break by replacing newlines with spaces
    nice_names = {k: v.replace("\n", " ") for k, v in nice_names_break.items()}
    
    return (
        datasets,
        quantities,
        nice_names,
        nice_names_break
    )


def clarke_ratio(A, a, B, b, C=100, c=0):
    """
    Calculate a ratio with error propagation.
    If A:B is the ratio, we want x where A:B = C:x
    """
    value = (B / A) * C
    uncertainty = np.sqrt((B * C * a / A) ** 2 + (C * b) ** 2 + (B * c) ** 2) / A
    return value, uncertainty


def clarke_quotient(A, a, B, b):
    """
    Get the ratio B:A and its error.
    """
    value = B / A
    uncertainty = np.sqrt((B * a / A) ** 2 + b**2) / A
    return value, uncertainty


def load_historical_data():
    """
    Load all historical datasets, truths, and nice names.
    
    Returns:
        tuple: (datasets, truths, nice_names)
    """
    # Load datasets
    c_df = pd.read_csv("data/c.csv", comment="#")
    
    c_oldford_df = pd.read_csv("data/c-oldford.csv", comment="#")
    c_oldford_df['value'] = c_oldford_df['speed']
    c_oldford_df['uncertainty'] = c_oldford_df['error'] * (1/0.6745)
    
    rho_df = pd.read_csv("data/rho.csv", comment="#")
    
    au_df = pd.read_csv("data/au.csv", comment="#")
    
    avogadro_df = pd.read_csv("data/bailey/CsvData/Constants_Data/Avogadro_Number.csv")
    # remove last row
    avogadro_df = avogadro_df.iloc[:-1]
    
    finestructure_df = pd.read_csv(
        "data/bailey/CsvData/Constants_Data/Fine_Structure_Constant_Inverse.csv"
    )
    
    rydberg_df = pd.read_csv("data/bailey/CsvData/Constants_Data/Rydberg_Constant.csv")
    
    c_bailey_df = pd.read_csv("data/bailey/CsvData/Constants_Data/Speed_of_Light.csv")
    
    datasets = {
        "rho": rho_df,
        "c": c_df,
        "au": au_df,
        "avogadro": avogadro_df,
        "finestructure": finestructure_df,
        "rydberg": rydberg_df,
        'c_bailey': c_bailey_df,
        "c_oldford": c_oldford_df,
    }
    
    # Define truths
    truths = {
        "rho": 5.513,
        "c": 299792.458,
        "au": 149597870700,
        "avogadro": 6.02214076e23,
        "finestructure": [137.035999177, 0.000000021],
        "rydberg": [10973731.568157, 0.000012],
        "c_bailey": 299792.458,
        "c_oldford": 299792.458,
    }
    
    # Define nice names
    nice_names = {
        "c": "Speed of light",
        "rho": "Density of Earth",
        "au": "Astronomical Unit",
        "avogadro": "Avogadro Number (Bailey)",
        "finestructure": r"$\alpha^{-1}$ (Bailey)",
        "rydberg": "Rydberg Constant (Bailey)",
        'c_bailey': 'Speed of light (Bailey)',
        "c_oldford": "Speed of light (Oldford)",
    }
    
    return datasets, truths, nice_names


def load_chemistry_data():
    """
    Load all chemistry datasets, truths, and nice names.
    
    Returns:
        tuple: (datasets, truths, nice_names)
    """
    # Load datasets
    ho_df = pd.read_csv("data/clarke/H-O-mass.csv", comment="#")
    ho_df["uncertainty"] = ho_df["proberr"] / 0.6745
    
    agcl_df = pd.read_csv("data/clarke/Ag-Cl-mass.csv", comment="#")
    agcl_df["uncertainty"] = agcl_df["proberr"] / 0.6745
    
    agi_df = pd.read_csv("data/clarke/Ag-I-mass.csv", comment="#")
    agi_df["uncertainty"] = agi_df["proberr"] / 0.6745
    
    agbr_df = pd.read_csv("data/clarke/Ag-Br-mass.csv", comment="#")
    agbr_df["uncertainty"] = agbr_df["proberr"] / 0.6745
    
    # Commented out datasets
    # no_df = pd.read_csv("data/clarke/N-mass.csv", comment="#")
    # no_df["uncertainty"] = no_df["proberr"] / 0.6745
    # co_df = pd.read_csv("data/clarke/C-mass.csv", comment="#")
    # co_df["uncertainty"] = co_df["proberr"] / 0.6745
    
    datasets = {
        "ho": ho_df,
        "agcl": agcl_df,
        "agi": agi_df,
        "agbr": agbr_df,
        # "no": no_df,
        # "co": co_df,
    }
    
    # Define truths
    truths = {}
    truths["agcl"] = clarke_ratio(107.8682, 0.0002, 35.453, 0.004, 100, 0)
    truths["agbr"] = clarke_ratio(107.8682, 0.0002, 79.904, 0.003, 100, 0)
    truths["agi"] = clarke_ratio(107.8682, 0.0002, 126.90447, 0.00003, 100, 0)
    truths["ho"] = clarke_quotient(1.0080, 0.0002, 15.9995, 0.0005)
    # truths['no'] = clarke_ratio(15.9995, 0.0005, 14.007, 0.001, 16, 0)
    # truths['co'] = clarke_ratio(15.9995, 0.0005, 12.011, 0.002, 16, 0)
    
    # Define nice names
    nice_names = {
        "ho": "O:H mass ratio",
        "agcl": "Ag:Cl mass ratio",
        "agi": "Ag:I mass ratio",
        "agbr": "Ag:Br mass ratio",
        # "no": 'N mass (O=16)',
        # "co": 'C mass (O=16)',
    }
    
    return datasets, truths, nice_names


def load_particle_data():
    """
    Load all particle datasets, truths, nice names, units, and group mappings.
    
    Returns:
        tuple: (datasets, truths, nice_names, units, particle_keys, particle_group_map)
    """
    # Get particle keys
    particle_keys = sorted(
        [f.split(".")[0] for f in os.listdir("data/pdg1970") if f.endswith(".csv")]
    )
    
    # Load datasets, truths, nice_names, and units
    datasets = {}
    truths = {}
    nice_names = {}
    units = {}
    
    for p in particle_keys:
        path = f"data/pdg1970/{p}.csv"
        datasets[p] = pd.read_csv(path, comment="#")
        with open(path, "r") as f:
            lines = f.readlines()
            nice_names[p] = lines[0].strip("# ").strip("\n")
            units[p] = lines[1].strip("# ").strip("\n")
            value = float(lines[2].split(":")[1].strip())
            sigma = float(lines[3].split(":")[1].strip())
            truths[p] = (value, sigma)
    
    # Create particle group map
    particle_group_keys = {
        "T": "Lifetime",
        "MM": "Mag.~moment",
        "M": "Mass",
        "M-": "Mass",
        "DEL": "Decay param",
        "D": "Mass diff",
        "WR": "Branching rate",
        "W": "Width",
        "R": "Branching ratio",
        "L+E": "Decay param",
        "A": "Decay param",
        "F+-": "Decay param",
        "XI": "Decay param",
    }
    particle_group_map = {}
    remaining = set(particle_keys)
    
    for k, v in particle_group_keys.items():
        for p in list(remaining):
            if k in p[1:]:
                particle_group_map[p] = v
                remaining.remove(p)
    
    assert len(remaining) == 0, f"Unmapped particles: {remaining}"
    
    # Add group names to nice_names
    nice_names.update({v: v for v in particle_group_keys.values()})
    
    return datasets, truths, nice_names, units, particle_keys, particle_group_map