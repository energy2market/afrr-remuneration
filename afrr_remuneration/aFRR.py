import typing
from collections import defaultdict
import numpy as np
import pandas as pd
import logging
from typing import (
    Tuple,
    Optional,
    Dict,
    TypeVar,
    Literal,
)

from dataclasses import dataclass

logger = logging.getLogger(__name__)
try:
    from numba import jit
except:
    logger.warning("Numba not found.")

    # this replaces the jit decorator in case numba is not found
    # so that nothing breaks

    def jit(sig, pyfunc=None, **kwargs):
        def wrap(func):
            return func

        if pyfunc is not None:
            return wrap(pyfunc)
        else:
            return wrap


MOL_TYPE = TypeVar("MOL_TYPE")
Mols = Dict[Tuple[pd.Timestamp, pd.Timestamp], Dict[float, MOL_TYPE]]

TSO = Literal["TNT", "50H", "AMP", "TBW"]


@dataclass
class RemunResult:
    remun: Mols[pd.Series]
    penalty: Mols[pd.Series]
    acceptance: Mols[pd.Series]
    underfulfillment: Mols[pd.Series]
    weights: Mols[pd.Series]

    def to_dataframe(self) -> pd.DataFrame:
        flat = {}
        for name in self.__dataclass_fields__.keys():
            for (start, stop), mol in getattr(self, name).items():
                for price, series in mol.items():
                    flat[(name, (start, stop), price)] = series
        if len(flat) > 0:
            return pd.concat(flat, axis="columns")
        else:
            return pd.DataFrame()


T = TypeVar("T")


@dataclass
class PosNegResult:
    pos: T
    neg: T

    def sum_dataframe(self) -> pd.DataFrame:
        res = []
        for direction in self.__dataclass_fields__.keys():
            df = getattr(self, direction).to_dataframe()
            df = df.groupby(level=0, axis="columns").sum()
            df.columns = df.columns + f"_{direction}"
            res.append(df)
        return pd.concat(res, axis="columns")


def calc_accept_remun(
    production_data: pd.DataFrame,
    neg_mol: Mols,
    pos_mol: Mols,
    penalty_price: Optional[pd.Series] = None,
    remun_price: Optional[pd.Series] = None,
):
    """
    calculation of accepted affr energy and remuneration in pos and neg direction
    :param production_data:
    :param neg_mol:
    :param pos_mol:
    :param penalty_price:
    :param remun_price:
    :return:
    """

    if len(neg_mol) + len(pos_mol) > 0:
        # calculate upper and lower gradient and respective acceptance and tolerance limits
        calculations = calc_acceptance_tolerance_band(
            setpoint=production_data.target, measured=production_data.actual
        )

        # Determination of pool acceptance values and underfulfillment
        df_underful = calc_underfulfillment_and_account(
            setpoint=calculations.setpoint,
            measured=calculations.measured,
            upper_acceptance_limit=calculations.upper_acceptance_limit,
            lower_acceptance_limit=calculations.lower_acceptance_limit,
            lower_tolerance_limit=calculations.lower_tolerance_limit,
            upper_tolerance_limit=calculations.upper_tolerance_limit,
        )
        if penalty_price is None:
            penalty_price = pd.Series(index=calculations.index, data=0)
        remun = calc_remuneration(
            upper_acceptance_limit=calculations.upper_acceptance_limit,
            lower_acceptance_limit=calculations.lower_acceptance_limit,
            allocable_acceptance_pos=df_underful.allocable_acceptance_pos,
            allocable_acceptance_neg=df_underful.allocable_acceptance_neg,
            allocable_underfulfill_pos=df_underful.allocable_underfulfill_pos,
            allocable_underfulfill_neg=df_underful.allocable_underfulfill_neg,
            delta_product_change_reversal_point=calculations.delta_product_change_reversal_point,
            mols_neg=neg_mol,
            mols_pos=pos_mol,
            remun_price=remun_price,
            penalty_price=penalty_price,
        )

        pool_remuns = remun.sum_dataframe()
        pool_result = pd.concat([pool_remuns, df_underful], axis=1)

        pool_result["measured_neg"] = -production_data.actual[
            production_data.actual < 0
        ].fillna(0)
        pool_result["measured_pos"] = production_data.actual[
            production_data.actual > 0
        ].fillna(0)

        return pool_result.convert_dtypes(float)
    else:
        return pd.DataFrame()


def calc_acceptance_tolerance_band(
    setpoint: pd.Series,
    measured: pd.Series,
    tolerance_band: float = 0.05,
    start_timeidx: int = 0,
) -> pd.DataFrame:
    """
    . Calculation of gradients:
        - gradient_upper_acceptance_limit (g_oga, "Gradient upper acceptance limit")
        - gradient_lower_acceptance_limit (g_uga, "Gradient lower acceptance limit")
    . definition of the change rate of the upper (oga) and lower (uga) acceptance limits:
        - positive
        - Reaching the new setpoint after 300 seconds at the latest.
        - Reaction to setpoint change after 30 seconds at the latest
        - Minimum change rate of 1 MW/270 seconds

    :param setpoint: setpoint values (per second) from TSO
    :param measured: measured values (per second)
    :param tolerance_band: definition of tolerance percentage
    :param start_timeidx: start row of index(timestamp), default: 0

    :return: pd.Dataframe containing gradients, acceptance and tolerance limits:
                df.columns=['setpoint','measured',
               'gradient_upper_acceptance_limit', 'gradient_lower_acceptance_limit',
               'upper_acceptance_limit', 'lower_acceptance_limit',
               'upper_tolerance_limit','lower_tolerance_limit']
    """
    # check whether every second has data. If not raise value error
    deltaT = setpoint.index[-1] - setpoint.index[0]
    duration_seconds = int(deltaT.total_seconds())
    diffE = (len(setpoint) - 1) - duration_seconds

    if len(setpoint) - 1 == duration_seconds:
        logging.info(
            f"OK. Every second has an entry! Duration: {duration_seconds} seconds"
        )
    else:
        raise ValueError(
            f"MISSING DATA! Not every second has an entry. (Difference = {diffE}). Check input!"
        )

    # define Series of setpoint and measurement values from a certain start index
    setpoint = round(setpoint.iloc[start_timeidx:], 3)
    measured = round(measured.iloc[start_timeidx:], 3)

    # print information about number of retrievals (setpoint != 0)
    logging.info(f"{np.count_nonzero(setpoint)} x nonzero in setpoint time series.")

    df = pd.DataFrame(
        index=setpoint.index,
        data={"setpoint": setpoint, "measured": measured},
        columns=[
            "setpoint",
            "measured",
            "gradient_lower_acceptance_limit",
            "gradient_upper_acceptance_limit",
            "lower_acceptance_limit",
            "upper_acceptance_limit",
            "lower_tolerance_limit",
            "upper_tolerance_limit",
            "is_product_change",
            "delta_product_change_reversal_point",
        ],
    )
    # initializations
    df.loc[df.index[0], "delta_product_change_reversal_point"] = 0
    df.loc[df.index[0], "upper_acceptance_limit"] = 0
    df.loc[df.index[0], "lower_acceptance_limit"] = 0

    # check whether t is time point of product change. works also during DST transition
    df.loc[:, "is_product_change"] = 0
    df.loc[
        (df.index.second == 0) & (df.index.minute == 0) & (df.index.hour % 4 == 0),
        "is_product_change",
    ] = 1

    # get signum of setpoint series
    signum_setpoint = np.sign(setpoint)

    # get minimum and maximum of setpoint values between t-31:t
    max_31_0 = setpoint.rolling(32, min_periods=1).max()
    min_31_0 = setpoint.rolling(32, min_periods=1).min()
    max_301_31 = setpoint.shift(periods=31).rolling(271, min_periods=1).max()
    min_301_31 = setpoint.shift(periods=31).rolling(271, min_periods=1).min()

    # calculate gradient of lower acceptance limit(t), formula 2
    df["gradient_lower_acceptance_limit"] = (
        (min_301_31 - min_31_0).abs().clip(lower=1) / 270
    ).round(3)

    # calculate gradient of upper Acceptance limit(t), formula 1
    # maximum of setpoint between t-301:t-31 (inclusive)
    df["gradient_upper_acceptance_limit"] = (
        (max_301_31 - max_31_0).abs().clip(lower=1) / 270
    ).round(3)

    # delta_product_change_reversal_point: difference of time between product change and reversal point
    # initialization
    df.delta_product_change_reversal_point = 0
    # iterate through each product change and find the reversal point afterwards ...
    # has to fulfill criteria of reversal point (bullet points 1.-5. on pages 13-14)
    for t in df.loc[df.is_product_change.astype(bool)].index:
        t_0 = t
        t_1 = t_0 + pd.Timedelta("1s")
        assert (
            t_0 - pd.Timedelta("1s")
        ) in setpoint, "setpoint must start before product change"
        if t_1 not in df.index:
            break
        # check whether already 300s passed since the product change / start of a new Leistungsscheibe
        logging.debug((df.delta_product_change_reversal_point[t_0] == 301))
        # check whether all setpoints within the following 65 seconds are not below the current setpoint
        logging.debug(
            (
                setpoint[t_0 : t_0 + pd.Timedelta("66s")].abs().min()
                >= abs(setpoint[t_0])
            )
        )
        # check whether setpoint crosses 0
        logging.debug(signum_setpoint[t_0] != signum_setpoint[t_1])
        # check whether setpoint is equal to 0
        logging.debug(setpoint[t_0] == 0)

        # the reversal point is reached and the product change phase is thus ended
        # if one of the following conditions is fulfilled (p. 16-17 in model description):
        while not (
            # maximum ramp duration is reached
            (df.delta_product_change_reversal_point[t_0] == 301)
            # all setpoints within the following 65 seconds are not below the current setpoint
            or (
                setpoint[t_0 : t_0 + pd.Timedelta("66s")].abs().min()
                >= abs(setpoint[t_0])
            )
            # setpoint changes signum (zero is crossed)
            or (
                signum_setpoint[t_0 - pd.Timedelta("1s")]
                != signum_setpoint[t_1 - pd.Timedelta("1s")]
            )
            # setpoint reaches zero
            or (setpoint[t_0] == 0)
        ):
            logging.debug(t_1, t_0, df.delta_product_change_reversal_point.loc[t_0])
            df.loc[t_1, "delta_product_change_reversal_point"] = (
                df.delta_product_change_reversal_point.loc[t_0] + 1
            )
            t_0 = t_1
            t_1 += pd.Timedelta("1s")

    tmp = _acceptance_tolerance_band(
        df.upper_acceptance_limit.values.astype(float),
        df.lower_acceptance_limit.values.astype(float),
        df.gradient_upper_acceptance_limit.values.astype(float),
        df.gradient_lower_acceptance_limit.values.astype(float),
        df.delta_product_change_reversal_point.values.astype(float),
        max_31_0.values.astype(float),
        min_31_0.values.astype(float),
    )

    (
        upper_acceptance_limit,
        lower_acceptance_limit,
        gradient_upper_acceptance_limit,
        gradient_lower_acceptance_limit,
    ) = tmp

    df.upper_acceptance_limit = upper_acceptance_limit
    df.lower_acceptance_limit = lower_acceptance_limit
    df.gradient_upper_acceptance_limit = gradient_upper_acceptance_limit
    df.gradient_lower_acceptance_limit = gradient_lower_acceptance_limit

    # 5% tolerance band: upper and lower limit (formula 5+6)
    df["upper_tolerance_limit"] = pd.to_numeric(
        df.upper_acceptance_limit + (abs(df.upper_acceptance_limit) * tolerance_band)
    ).round(3)

    df["lower_tolerance_limit"] = pd.to_numeric(
        df.lower_acceptance_limit - (abs(df.lower_acceptance_limit) * tolerance_band)
    ).round(3)

    return df


@jit(
    "Tuple((float64[:], float64[:], float64[:], float64[:]))(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:])",
    nopython=True,
)
def _acceptance_tolerance_band(
    upper_acceptance_limit,
    lower_acceptance_limit,
    gradient_upper_acceptance_limit,
    gradient_lower_acceptance_limit,
    delta_product_change_reversal_point,
    max_31_0,
    min_31_0,
):
    """
    inner loop of acceptance_tolerance_band
    :param upper_acceptance_limit:
    :param lower_acceptance_limit:
    :param gradient_upper_acceptance_limit:
    :param gradient_lower_acceptance_limit:
    :param delta_product_change_reversal_point:
    :param max_31_0:
    :param min_31_0:
    :return:
    """
    # calculate upper acceptance limit prerequisites:
    # maximum out of all setpoints between (t-31) -> (t) and
    # the upper acceptance limit of one time step before (t-1)-upper gradient at t
    for i in range(len(upper_acceptance_limit)):
        # calculate upper acceptance limit prerequisites
        max_upper_acceptance_limit = max(
            max_31_0[i],
            (upper_acceptance_limit[i - 1] - gradient_upper_acceptance_limit[i]),
        )

        # calculate lower acceptance limit prerequisite:
        # minimum out of all setpoints between (t-31) -> (t) and
        # the lower acceptance limit of one time step before (t-1)-lower gradient at t
        min_lower_acceptance_limit = min(
            min_31_0[i],
            (lower_acceptance_limit[i - 1] + gradient_lower_acceptance_limit[i]),
        )

        # calculate upper acceptance limit (formula 3)
        # if t_productchange > t > t_reversalpoint
        upper_acceptance_limit[i] = (
            max_upper_acceptance_limit
            if delta_product_change_reversal_point[i] == 0
            # else (t_productchange <= t <= t_reversalpoint)
            else max(max_upper_acceptance_limit, 0)
        )
        # calculate lower acceptance limit (formula 4)
        lower_acceptance_limit[i] = (
            min_lower_acceptance_limit
            if delta_product_change_reversal_point[i] == 0
            else min(min_lower_acceptance_limit, 0)
        )

    return (
        upper_acceptance_limit,
        lower_acceptance_limit,
        gradient_upper_acceptance_limit,
        gradient_lower_acceptance_limit,
    )


def calc_underfulfillment_and_account(
    setpoint: pd.Series,
    measured: pd.Series,
    upper_acceptance_limit: pd.Series,
    lower_acceptance_limit: pd.Series,
    lower_tolerance_limit: pd.Series,
    upper_tolerance_limit: pd.Series,
    tolerance_band: float = 0.05,
) -> pd.DataFrame:
    """
    Calculation of allocatable and not allocatable underfulfillment and acceptance values
    :param tolerance_band: definition of tolerance percentages (5 % as default)
    :param setpoint: setpoint values
    :param measured: measured values
    calculated via function 'acceptance_tolerance_band':
    :param upper_acceptance_limit: respective upper acceptance limits
    :param lower_acceptance_limit: respective lower acceptance limits
    :param upper_tolerance_limit: upper limits of tolerance band
    :param lower_tolerance_limit: lower limits of tolerance band

    :return: pd.DataFrame: df.columns=[
            "acceptance_pool_pos": "akz_pos",
            "acceptance_pool_neg":"akz_neg",
            "acceptance_value",
            "underfulfill_pool_pos":"ue_pos",
            "underfulfill_pool_neg":"ue_neg",
            "underfull_flag_pos": "ueflag_pos,
            "underfull_flag_neg":"ueflag_neg",
            "setpoint_pos": "s_pos",
            "setpoint_neg": "s_neg",
            "account_pos": "Konto_pos",
            "account_neg": "Konto_neg",
            "allocable_acceptance_pos": "zak_pos",
            "allocable_acceptance_neg":"zak_neg",
            "allocable_acceptance_value": "zak / zuteilbarer Akzeptanzwert",
            "allocable_underfulfill_pos":"zue_pos",
            "allocable_underfulfill_neg":"zue_neg",
            "allocable_underfulfill",
            "not_allocable_underfulfill": "zue_value_nicht_zuteilbar",
            "overfulfill_pos":"ueb_pos",
            "overfulfill_neg":"ueb_neg",
            "cutoff":"Kappung"]
    """
    df_underperf = pd.DataFrame(
        index=setpoint.index,
        columns=[
            "acceptance_pool_pos",
            "acceptance_pool_neg",
            "acceptance_value",
            "underfulfill_pool_pos",
            "underfulfill_pool_neg",
            "underfull_flag_pos",
            "underfull_flag_neg",
            "setpoint_pos",
            "setpoint_neg",
            "account_pos",
            "account_neg",
            "allocable_acceptance_pos",
            "allocable_acceptance_neg",
            "allocable_acceptance_value",
            "allocable_underfulfill_pos",
            "allocable_underfulfill_neg",
            "allocable_underfulfill",
            "not_allocable_underfulfill",
            "overfulfill_pos",
            "overfulfill_neg",
            "cutoff",
        ],
    )

    # initialize else = 0 of several if-conditions
    df_underperf.loc[:, ["acceptance_pool_pos", "acceptance_pool_neg"]] = 0
    df_underperf.loc[:, "underfulfill_pool_pos":"setpoint_neg"] = 0
    df_underperf.loc[:, ["overfulfill_pos", "overfulfill_neg"]] = 0
    df_underperf.loc[
        :, ["allocable_underfulfill_pos", "allocable_underfulfill_neg"]
    ] = 0

    # Pool acceptance values for the positive and negative direction
    # the positive acceptance value is the minimum of the measured value
    # and the upper acceptance limit if the measured value is positive
    # and upper acceptance limit is positive (formula 7)
    df_underperf.loc[
        (measured > 0) & (upper_acceptance_limit > 0),
        "acceptance_pool_pos",
    ] = np.minimum(measured, upper_acceptance_limit)
    # pool acceptance value for negative direction (formula 8)
    df_underperf.loc[
        (measured < 0) & (lower_acceptance_limit < 0),
        "acceptance_pool_neg",
    ] = abs(np.maximum(measured, lower_acceptance_limit))

    # pool underfulfillment values for positive and negative direction (formula 9+10)
    df_underperf.loc[(lower_tolerance_limit > 0), "underfulfill_pool_pos"] = np.maximum(
        0, lower_tolerance_limit - df_underperf.acceptance_pool_pos
    )

    df_underperf.loc[(upper_tolerance_limit < 0), "underfulfill_pool_neg"] = np.maximum(
        0, abs(upper_tolerance_limit) - df_underperf.acceptance_pool_neg
    )

    # calculation of allocable underfulfillment
    # preparation: determine flags for underfulfillment (formula 17 and 18)
    df_underperf.loc[(df_underperf.underfulfill_pool_pos > 0), "underfull_flag_pos"] = 1
    df_underperf.loc[(df_underperf.underfulfill_pool_neg > 0), "underfull_flag_neg"] = 1

    # condition that is > 0.05 implies underfulfillment (formula 19)
    cond_pos = df_underperf.underfull_flag_pos.rolling(300).sum().fillna(value=0) / 300
    cond_neg = df_underperf.underfull_flag_neg.rolling(300).sum().fillna(value=0) / 300

    # calculation of allocable underfulfillment (formula 19+20)
    df_underperf.loc[
        (cond_pos > tolerance_band), "allocable_underfulfill_pos"
    ] = df_underperf.underfulfill_pool_pos

    df_underperf.loc[
        (cond_neg > tolerance_band), "allocable_underfulfill_neg"
    ] = df_underperf.underfulfill_pool_neg

    # calculate positive and negative setpoint components (formula 11+12)
    df_underperf["setpoint_pos"] = np.maximum(0, setpoint)
    df_underperf["setpoint_neg"] = abs(np.minimum(0, setpoint))

    tmp = _calculate_underfulfillment_and_account(
        df_underperf.allocable_acceptance_pos.values.astype(float),
        df_underperf.allocable_acceptance_neg.values.astype(float),
        upper_acceptance_limit.values.astype(float),
        lower_acceptance_limit.values.astype(float),
        df_underperf.setpoint_pos.values.astype(float),
        df_underperf.setpoint_neg.values.astype(float),
        df_underperf.account_pos.values.astype(float),
        df_underperf.account_neg.values.astype(float),
        df_underperf.acceptance_pool_pos.values.astype(float),
        df_underperf.acceptance_pool_neg.values.astype(float),
    )

    account_pos, account_neg, allocable_acceptance_pos, allocable_acceptance_neg = tmp
    df_underperf.account_pos = account_pos
    df_underperf.account_neg = account_neg
    df_underperf.allocable_acceptance_pos = allocable_acceptance_pos
    df_underperf.allocable_acceptance_neg = allocable_acceptance_neg

    # else: not allocable underfulfillment [MW]
    df_underperf["not_allocable_underfulfill"] = -(
        df_underperf.underfulfill_pool_neg - df_underperf.allocable_underfulfill_neg
    )
    # if...
    df_underperf.loc[
        (df_underperf.underfulfill_pool_pos > 0), "not_allocable_underfulfill"
    ] = (df_underperf.underfulfill_pool_pos - df_underperf.allocable_underfulfill_pos)

    # allocable underfulfillment [MW]
    df_underperf["allocable_underfulfill"] = (
        df_underperf.allocable_underfulfill_pos
        - df_underperf.allocable_underfulfill_neg
    )

    # Overfulfillment = not allocable acceptance value (formula 21+22 in "old" documentation of 2021-05-06):
    # part of the measured values which cant be accounted
    df_underperf.loc[(measured >= 0), "overfulfill_pos"] = (
        measured - df_underperf.allocable_acceptance_pos
    )
    df_underperf.loc[(measured < 0), "overfulfill_neg"] = (
        abs(measured) - df_underperf.allocable_acceptance_neg
    )

    # acceptance value [MW]: difference of acceptance value of pos and neg direction
    df_underperf["acceptance_value"] = (
        df_underperf.acceptance_pool_pos - df_underperf.acceptance_pool_neg
    )
    # allocable acceptance value (zuteilbarer Akzeptanzwert) [MW]
    df_underperf["allocable_acceptance_value"] = (
        df_underperf.allocable_acceptance_pos - df_underperf.allocable_acceptance_neg
    )

    # calculate cutoff ("Kappung")
    df_underperf["cutoff"] = (
        (df_underperf.acceptance_value - df_underperf.allocable_acceptance_value)
        .astype(float)
        .round(8)
    )

    return df_underperf


@jit(
    "Tuple((float64[:], float64[:], float64[:], float64[:]))"
    + "(float64[:], float64[:], float64[:], float64[:], float64[:],"
    + " float64[:], float64[:], float64[:], float64[:], float64[:])",
    nopython=True,
)
def _calculate_underfulfillment_and_account(
    allocable_acceptance_pos,
    allocable_acceptance_neg,
    upper_acceptance_limit,
    lower_acceptance_limit,
    setpoint_pos,
    setpoint_neg,
    account_pos,
    account_neg,
    acceptance_pool_pos,
    acceptance_pool_neg,
):
    """
    inner loop of calculate_underfulfillment_and_account
    :param allocable_acceptance_pos:
    :param allocable_acceptance_neg:
    :param upper_acceptance_limit:
    :param lower_acceptance_limit:
    :param setpoint_pos:
    :param setpoint_neg:
    :param account_pos:
    :param account_neg:
    :param acceptance_pool_pos:
    :param acceptance_pool_neg:
    :return:
    """
    for i in range(len(allocable_acceptance_pos)):
        if i == 0:
            continue
        # allocable acceptance values for positive direction (formula 13)
        allocable_acceptance_pos[i] = round(
            min(
                (setpoint_pos[i] + account_pos[i - 1]),
                acceptance_pool_pos[i],
            ),
            3,
        )
        # allocable acceptance values for negative direction (formula 14)
        allocable_acceptance_neg[i] = round(
            min(
                setpoint_neg[i] + account_neg[i - 1],
                acceptance_pool_neg[i],
            ),
            3,
        )

        # calculate account: sums up the difference resulting in each calculation interval that the
        # allocable acceptance value is below the setpoint. Its maximum is the difference from the inner
        # acceptance channel limit to the setpoint.
        # -> avoids the incentive to make up the missing energy at the end of the aFRR-call

        # positive account (formula 15)
        # preparation: get maximum between allocable acceptance value(pos), 0, lower acceptance limit
        max_zakP_lower_acceptance_limit = max(
            allocable_acceptance_pos[i],
            max(0, lower_acceptance_limit[i]),
        )

        account_pos[i] = (
            max(
                0,
                setpoint_pos[i] - max_zakP_lower_acceptance_limit + account_pos[i - 1],
            )
            if upper_acceptance_limit[i] > 0
            else 0
        )

        # negative account (formula 16)
        # preparation: get max of allocable acceptance (neg) and the absolute min of 0 and upper acceptance limit
        max_zakN_upper_acceptance_limit = max(
            allocable_acceptance_neg[i],
            abs(min(0, upper_acceptance_limit[i])),
        )
        account_neg[i] = (
            max(
                0,
                setpoint_neg[i] - max_zakN_upper_acceptance_limit + account_neg[i - 1],
            )
            if lower_acceptance_limit[i] < 0
            else 0
        )

    return account_pos, account_neg, allocable_acceptance_pos, allocable_acceptance_neg


def calc_bid_weights(
    mols: Mols[float], acc_limit: pd.Series, is_ramp: pd.Series
) -> Mols[pd.Series]:
    """
    calculates the weight for each bid in the mol given the acceptance limit and ramp.

    The proper accepance limit for the mol pos / neg -> upper / lower must be supplied. The ramp is used to shorten the
    valid time for each bid, after a product change and elongate the time of a bid past a product change
    :param mols: mols for the direction (see calc_remuneration for an example)
    :param acc_limit: upper/lower_acceptance_limit
    :return:
    """
    weights = {}
    for (start, stop), mol in mols.items():
        # the lower limit of the range within the mol
        # is defined by the upper limit of the previous entry
        # except for the first one where it is 0

        t_mask = acc_limit.index >= start
        t_mask &= acc_limit.index < stop
        t_mask = pd.Series(t_mask, index=acc_limit.index)

        if not t_mask.any():
            continue
        # the ramp needs to be added/substracted but only if it is within the valid timeframe of
        # the mol + 5min
        t_mask[start : stop + pd.Timedelta("300s")] ^= is_ramp[
            start : stop + pd.Timedelta("300s")
        ]

        mol = pd.DataFrame.from_dict(mol, orient="index", columns=["volume"])
        mol.sort_index(inplace=True)
        mol.volume = mol.volume.cumsum()
        mol["lower"] = mol.volume.shift(periods=1).fillna(value=0)

        weights[(start, stop)] = {}

        for price, (upper, lower) in mol.iterrows():
            weight = pd.Series(index=acc_limit.index, data=0)

            between_mask = acc_limit.between(lower, upper)
            above_mask = acc_limit > upper
            n0_mask = acc_limit != 0

            weight[t_mask & above_mask & n0_mask] = (upper - lower) / acc_limit[
                t_mask & above_mask & n0_mask
            ]
            weight[t_mask & between_mask & n0_mask] = (
                acc_limit[t_mask & between_mask & n0_mask] - lower
            ) / acc_limit[t_mask & between_mask & n0_mask].abs()

            weights[(start, stop)][price] = weight.round(8)
    return weights


def calc_remuneration(
    upper_acceptance_limit: pd.Series,
    lower_acceptance_limit: pd.Series,
    allocable_acceptance_pos: pd.Series,
    allocable_acceptance_neg: pd.Series,
    allocable_underfulfill_pos: pd.Series,
    allocable_underfulfill_neg: pd.Series,
    delta_product_change_reversal_point: pd.Series,
    remun_price: pd.Series,
    penalty_price: pd.Series,
    mols_pos: Mols[float],
    mols_neg: Mols[float],
) -> PosNegResult:
    """
    Convenience function to calculate weights, remuneration and penalty for each bid in the positive and negative mol.
    Every mol is assumed to be of the structure:
    >>> mol = {
    >>>     (start, stop):
    >>>     {
    >>>         price1: value1,
    >>>         price2: value2
    >>>     },
    >>>     (start1, stop2):
    >>>     {
    >>>         price3: value3,
    >>>         price4: value4
    >>>     }
    >>>}
    Where start and stop represent the duration where in which each mol is valid, while the prices are the steps in
    the merit order.
    For the mols_pos and mols_neg the values are the size of the accepted bid in MW. For the results the respective
    pandas Series will be placed in there.
    :param upper_acceptance_limit: upper_acceptance_limit
    :param lower_acceptance_limit: lower_acceptance_limit
    :param allocable_acceptance_pos: allocable_acceptance_pos
    :param allocable_acceptance_neg: allocable_acceptance_neg
    :param allocable_underfulfill_pos: allocable_underfulfill_pos
    :param allocable_underfulfill_neg: allocable_underfulfill_neg
    :param remun_price: price to use for remuneration
    :param penalty_price: price to use for penalty
    :param mols_pos: mols for pos bids
    :param mols_neg: mols for neg bids
    :return:
    """
    weights_pos = calc_bid_weights(
        mols=mols_pos,
        acc_limit=upper_acceptance_limit,
        is_ramp=(delta_product_change_reversal_point > 0),
    )

    allocable_underfulfill_pos = allocable_underfulfill_pos.astype(float)
    allocable_underfulfill_neg = allocable_underfulfill_neg.astype(float)

    pos = apply_weights(
        allocable_acceptance=allocable_acceptance_pos,
        allocable_underfulfill=allocable_underfulfill_pos,
        penalty_price=penalty_price,
        remun_price=remun_price,
        weights=weights_pos,
        price_direction="pos",
    )

    weights_neg = calc_bid_weights(
        mols=mols_neg,
        acc_limit=-lower_acceptance_limit,
        is_ramp=(delta_product_change_reversal_point > 0),
    )

    neg = apply_weights(
        allocable_acceptance=allocable_acceptance_neg,
        allocable_underfulfill=allocable_underfulfill_neg,
        penalty_price=penalty_price,
        remun_price=remun_price,
        weights=weights_neg,
        price_direction="neg",
    )

    return PosNegResult(pos=pos, neg=neg)


def apply_weights(
    allocable_acceptance: pd.Series,
    allocable_underfulfill: pd.Series,
    penalty_price: pd.Series,
    remun_price: pd.Series,
    weights: pd.Series,
    price_direction: typing.Literal["pos", "neg"],
):
    penalty = defaultdict(dict)
    remun = defaultdict(dict)
    acceptance = defaultdict(dict)
    underfulfillment = defaultdict(dict)
    for start_stop, mol in weights.items():
        for price, weight in mol.items():

            if remun_price is not None:
                tmp_price = remun_price.copy()
                if price_direction == "pos":
                    tmp_price[tmp_price < price] = price
                else:
                    tmp_price[tmp_price < price] = price
            else:
                tmp_price = price

            underfulfillment[start_stop][price] = (
                allocable_underfulfill * weight
            ).round(3)
            penalty[start_stop][price] = (
                underfulfillment[start_stop][price] * penalty_price / 3600
            )
            acceptance[start_stop][price] = (allocable_acceptance * weight).round(3)
            remun[start_stop][price] = acceptance[start_stop][price] * tmp_price / 3600
            # ensure penalty is alway < 0
            p = penalty[start_stop][price]
            penalty[start_stop][price][p > 0] *= 1

    res = RemunResult(
        remun=remun,
        penalty=penalty,
        acceptance=acceptance,
        underfulfillment=underfulfillment,
        weights=weights,
    )
    return res


def mol2df(mols: Mols[pd.DataFrame], multiindex=False) -> pd.DataFrame:
    """
    flattens a mol into a dataframe with the columns containing the start, stop and price
    :param mols: mols to transform
    :return:
    """
    if multiindex:
        flat = {
            ((start, stop), price): series
            for (start, stop), mol in mols.items()
            for price, series in mol.items()
        }
    else:
        flat = {
            f"{start} -> {stop}: {price}": series
            for (start, stop), mol in mols.items()
            for price, series in mol.items()
        }
    return pd.concat(flat, axis="columns")
