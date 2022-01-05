import pytest
import pandas as pd

from aFRR import calc_acceptance_tolerance_band, calc_underfulfillment_and_account
from data import parse_tso_data
from pathlib import Path


@pytest.fixture(
    scope="module",
    params=[
        "20211231_aFRR_XXXXXXXXXXXXXXXX_XXX_PT1S_043_V01.csv",
    ],
)
def sec_test_data(request):
    examples = Path("example_data/")
    file = examples / request.param
    pool_df, bids = parse_tso_data(file)
    return pool_df


def test_sec_validation_results(sec_test_data):
    band_df = calc_acceptance_tolerance_band(
        setpoint=sec_test_data["setpoint"], measured=sec_test_data["measured"]
    )

    underful_df = calc_underfulfillment_and_account(
        setpoint=band_df.setpoint,
        measured=band_df.measured,
        upper_acceptance_limit=band_df.upper_acceptance_limit,
        lower_acceptance_limit=band_df.lower_acceptance_limit,
        lower_tolerance_limit=band_df.lower_tolerance_limit,
        upper_tolerance_limit=band_df.upper_tolerance_limit,
    )

    columns = sec_test_data.columns.intersection(underful_df.columns)

    pd.testing.assert_frame_equal(
        sec_test_data[columns].iloc[2:],
        underful_df[columns].iloc[2:],
        check_names=False,
        check_dtype=False,
        rtol=0.01,
        atol=0.1,
    )
