from pathlib import Path
from typing import Optional, Union, Dict
import pandas as pd
from aFRR import Mols

TRANSLATION_POOL_VALUES = {
    "AKZ_NEG": "acceptance_pool_neg",
    "AKZ_POS": "acceptance_pool_pos",
    "EIST_NEGPOS": "replacements_measured",
    "ESOLL_NEGPOS": "replacements_setpoint",
    "IST_NEG": "measured_neg",
    "IST_POS": "measured_pos",
    "KZAK_NEG": "renum_neg",
    "KZAK_POS": "renum_pos",
    "KZUE_NEG": "penalty_neg",
    "KZUE_POS": "penalty_pos",
    "SOLL_NEG": "setpoint_neg",
    "SOLL_POS": "setpoint_pos",
    "UEB_NEG": "overfulfill_neg",
    "UEB_POS": "overfulfill_pos",
    "UE_NEG": "underfulfill_pool_neg",
    "UE_POS": "underfulfill_pool_pos",
    "ZAK_NEG": "allocable_acceptance_neg",
    "ZAK_POS": "allocable_acceptance_pos",
    "ZUE_NEG": "allocable_underfulfill_neg",
    "ZUE_POS": "allocable_underfulfill_pos",
}


def parse_tso_data(
    file: Union[str, Path], bid_ids: Optional[Mols[str]] = None
) -> (pd.DataFrame, Union[Mols[pd.DataFrame], Dict[str, pd.DataFrame]]):

    df = pd.read_csv(file, index_col=0, sep=";", decimal=",").T
    df.index = pd.DatetimeIndex(df.index)

    file = Path(file)
    filename = file.name.replace(".csv", "")
    date_, r_type, ID, TSO_, freq_, offest, version = filename.split("_")

    raw_id = pd.Series(df.columns).str.split("_", expand=True)
    raw_id.index = df.columns
    raw_id[2] = raw_id[2].str.replace("SRA", "")
    pool = raw_id.loc[raw_id[0] == ID, :].copy()
    bid = raw_id.loc[raw_id[0] != ID, :].copy()
    del raw_id
    pool["name"] = pool[3] + "_" + pool[2]
    bid["name"] = bid[3] + "_" + bid[2]
    bid.name = bid.name.apply(lambda x: TRANSLATION_POOL_VALUES[x])
    pool.name = pool.name.apply(lambda x: TRANSLATION_POOL_VALUES[x])

    pool_df = df[pool.index].rename(pool.name, axis=1)
    # make mol with bid data
    bids = {}
    if bid_ids is not None:
        for timestamps in bid_ids:
            bids[timestamps] = {}
            for price in bid_ids[timestamps]:
                ind = bid.loc[bid[0] == bid_ids[timestamps][price]].index
                bids[timestamps][price] = df[ind].rename(bid.name, axis=1)
    else:
        for id in bid[0].unique():
            ind = bid.loc[bid[0] == id].index
            bids[id] = df[ind].rename(bid.name, axis=1)

    # combine pos and neg setpoint and measured
    pool_df["setpoint"] = pool_df["setpoint_pos"] - pool_df["setpoint_neg"]
    pool_df["measured"] = pool_df["measured_pos"] - pool_df["measured_neg"]

    # convert the volumes to power values
    if freq_ == "PT1S":
        pool_df["allocable_acceptance_neg"] *= 3600 
        pool_df["allocable_acceptance_pos"] *= 3600
    elif freq_ == "PT15M":
        pool_df["allocable_acceptance_neg"] *= 4
        pool_df["allocable_acceptance_pos"] *= 4
    else:
        raise ValueError(f"Unsupported frequency in tso data. Only PT1S and PT15M supported. Given {freq_}")

    return pool_df, bids
