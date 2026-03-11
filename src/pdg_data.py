import sqlite3
import pandas as pd
import numpy as np
from collections import defaultdict

PDG_QUERY = """
SELECT pdgid.description, pdgmeasurement.pdgid, pdgmeasurement_values.column_name, pdgdata.value_type, pdgmeasurement.technique, pdgreference.publication_year, pdgid.data_type, pdgdata.in_summary_table, pdgdata.value, pdgdata.error_negative, pdgdata.error_positive, pdgmeasurement_values.value, pdgmeasurement_values.error_positive, pdgmeasurement_values.error_negative, pdgmeasurement_values.stat_error_positive, pdgmeasurement_values.stat_error_negative, prev.value, prev.error_negative, prev.error_positive, prev.edition
FROM pdgmeasurement_values
    JOIN pdgmeasurement ON pdgmeasurement.id = pdgmeasurement_values.pdgmeasurement_id
    JOIN pdgid ON pdgid.id = pdgmeasurement.pdgid_id
    JOIN pdgdata ON pdgdata.pdgid_id = pdgid.id
    JOIN pdgreference ON pdgreference.id = pdgmeasurement.pdgreference_id
--     JOIN pdgparticle ON pdgparticle.pdgid = pdgid.parent_pdgid
    LEFT JOIN pdgdata AS prev
        ON prev.id = (
            SELECT d.id
            FROM pdgdata d
            WHERE d.pdgid_id = pdgid.id
--            AND d.value_type = pdgdata.value_type
            AND CAST(d.edition AS INTEGER) < CAST(pdgreference.publication_year AS INTEGER)
            ORDER BY CAST(d.edition AS INTEGER) DESC
            LIMIT 1
        )
WHERE {avg_param} pdgmeasurement_values.used_in_average AND pdgmeasurement_values.value IS NOT NULL AND pdgdata.edition = '2025' AND pdgdata.value_type = 'AC'
"""


def load_pdg_data(used_in_average=True):
    con = sqlite3.connect("../data/pdgall-2025-v0.2.1.sqlite")
    cur = con.cursor()
    avg_param = "" if used_in_average else "NOT"
    res = cur.execute(PDG_QUERY.format(avg_param=avg_param))
    data = res.fetchall()
    columns = [col[0] for col in res.description]
    print(len(data), "measurements")
    print(columns)

    df = pd.DataFrame(
        data,
        columns=[
            "pdgid.description",
            "pdgid",
            "column_name",
            "type",
            "technique",
            "year",
            "data_type",
            "insummary",
            "avg",
            "avg_error_negative",
            "avg_error_positive",
            "measurement",
            "error_positive",
            "error_negative",
            "stat_error_positive",
            "stat_error_negative",
            "prev_value",
            "prev_error_negative",
            "prev_error_positive",
            "prev_edition",
        ],
    )
    df = df[df["column_name"] == "VALUE"]
    df["error"] = (df["error_positive"] + df["error_negative"]) / 2
    # drop rows where error is NaN
    df = df.dropna(subset=["error"])
    # replace NaN staterr with error
    df["stat_error_negative"] = df["stat_error_negative"].fillna(df["error_negative"])
    df["stat_error_positive"] = df["stat_error_positive"].fillna(df["error_positive"])
    # see https://pdg.lbl.gov/encoder_listings/s035.pdf, S035C19 LEE, 0.003% stat. error * 8.42
    # staterr[(staterr == 0) & np.array(df["pdgid"] == "S035C19")] = 0.0002526
    df.loc[
        (df["stat_error_negative"] == 0) & (df["pdgid"] == "S035C19"),
        "stat_error_negative",
    ] = 0.0002526
    df.loc[
        (df["stat_error_positive"] == 0) & (df["pdgid"] == "S035C19"),
        "stat_error_positive",
    ] = 0.0002526

    df["staterr"] = (df["stat_error_positive"] + df["stat_error_negative"]) / 2
    # replace NaN staterr with total error
    # staterr = np.array(df["staterr"])
    # staterr[np.isnan(staterr)] = np.array(df["error"])[np.isnan(staterr)]
    # replace zero staterr with smallest nonzero staterr
    # staterr[staterr == 0] = np.min(staterr[staterr > 0])/2
    # df["staterr"] = staterr

    df["std_resid"] = (df["measurement"] - df["avg"]) / df["error"]
    # only keep rows where there are at least 2 measurements
    df = df.groupby("pdgid").filter(lambda x: len(x) >= 2)

    df["value"] = df["measurement"]
    # to_drop = ['measurement', 'error_positive', 'error_negative', 'stat_error_positive', 'stat_error_negative']
    # df = df.drop(columns=to_drop)

    print("Number of properties:", len(df["pdgid"].unique()))
    print("Number of measurements:", len(df))

    pdg2025_stat = df.copy()
    pdg2025_stat["uncertainty"] = pdg2025_stat["staterr"]
    del pdg2025_stat["staterr"], pdg2025_stat["error"]
    df_gb = pdg2025_stat.groupby("pdgid", group_keys=False)
    pdg2025_stat_dfs = [
        df_gb.get_group(x).copy()
        for x in df_gb.groups
        if all(df_gb.get_group(x)["uncertainty"] > 0)
    ]

    pdg2025_both = df.copy()
    pdg2025_both["uncertainty"] = pdg2025_both["error"]
    del pdg2025_both["staterr"], pdg2025_both["error"]
    df_gb = pdg2025_both.groupby("pdgid", group_keys=False)
    pdg2025_both_dfs = [df_gb.get_group(x).copy() for x in df_gb.groups]

    pdg2025_stat_quantities = [df["pdgid"].iloc[0] for df in pdg2025_stat_dfs]
    pdg2025_both_quantities = [df["pdgid"].iloc[0] for df in pdg2025_both_dfs]

    return (
        pdg2025_stat_dfs,
        pdg2025_both_dfs,
        pdg2025_stat_quantities,
        pdg2025_both_quantities,
    )


def pdgid_type_map():
    con = sqlite3.connect("../data/pdgall-2025-v0.2.1.sqlite")
    cur = con.cursor()
    DATA_TYPE_QUERY = """
    SELECT pdgid.pdgid, pdgid.data_type, pdgdoc.description
    FROM pdgid
    JOIN pdgdoc ON pdgid.data_type = pdgdoc.value
    WHERE pdgdoc.table_name = 'PDGID' AND pdgdoc.column_name = 'DATA_TYPE'
    """
    res = cur.execute(DATA_TYPE_QUERY)
    data = res.fetchall()
    type_map = defaultdict(lambda: "Other")
    for row in data:
        type_map[row[0]] = row[2]
    return type_map
