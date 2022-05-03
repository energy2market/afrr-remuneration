import pytest
import pandas as pd
from afrr_remuneration.aFRR import *
from afrr_remuneration.data import *

@pytest.fixture(scope="module")
def load_data():
    
    # Load TSO awarded capacity data from example data
    awarded_pos_capacity = pd.read_csv("example_data/awarded_pos_capacity.csv",  parse_dates=[0], index_col=0).squeeze()
    awarded_neg_capacity = pd.read_csv("example_data/awarded_neg_capacity.csv",  parse_dates=[0], index_col=0).squeeze()
    
    # Load TSO datasets for hours 2 - 24 from example data
    tso_data = dict()
    for h in range(2, 25):
        tso_data[h] = pd.read_csv(f"example_data/tso_data_H{str(h)}.csv",  parse_dates=[0], index_col=0)
        tso_data[h]["is_product_change_tso"] = tso_data[h]["is_product_change_tso"].notna().astype(int)
        
        # Execute calculation functions on datasets
        band_data = calc_acceptance_tolerance_band(
            setpoint=tso_data[h]["setpoint_tso"], 
            measured=tso_data[h]["measured_tso"],
            awarded_pos_capacity=awarded_pos_capacity,
            awarded_neg_capacity=awarded_neg_capacity,
        )
        tso_data[h] = tso_data[h].join(band_data)
        fulfillment_data = calc_underfulfillment_and_account(
            setpoint=tso_data[h]["setpoint_tso"],
            measured=tso_data[h]["measured_tso"],
            upper_acceptance_limit=tso_data[h]["upper_acceptance_limit"],
            lower_acceptance_limit=tso_data[h]["lower_acceptance_limit"],
            upper_tolerance_limit=tso_data[h]["upper_tolerance_limit"],
            lower_tolerance_limit=tso_data[h]["lower_tolerance_limit"],
        ).astype(float)
        tso_data[h] = tso_data[h].join(fulfillment_data)
        
        # Remove data of first 602 seconds (stabilization time)
        tso_data[h] = tso_data[h].iloc[602:, :]
    return tso_data
    
@pytest.mark.parametrize(
    "s1, s2, rtol, atol",
    [
        pytest.param(
            "gradient_lower_acceptance_limit", 
            "gradient_lower_acceptance_limit_tso", 
            0.001, 
            0.001, 
            id="gradient_lower_acceptance_limit",
        ),
        pytest.param(
            "gradient_upper_acceptance_limit", 
            "gradient_upper_acceptance_limit_tso", 
            0.001, 
            0.001, 
            id="gradient_upper_acceptance_limit",
        ),
        pytest.param(
            "lower_acceptance_limit", 
            "lower_acceptance_limit_tso", 
            0.001, 
            0.001, 
            id="lower_acceptance_limit",
        ),
        pytest.param(
            "upper_acceptance_limit", 
            "upper_acceptance_limit_tso", 
            0.001, 
            0.001, 
            id="upper_acceptance_limit",
        ),
        pytest.param(
            "lower_tolerance_limit", 
            "lower_tolerance_limit_tso", 
            0.001, 
            0.00105, 
            id="lower_tolerance_limit",
        ),
        pytest.param(
            "upper_tolerance_limit", 
            "upper_tolerance_limit_tso", 
            0.001, 
            0.00105, 
            id="upper_tolerance_limit",
        ),
       pytest.param(
            "is_product_change", 
            "is_product_change_tso", 
            0, 
            0, 
            id="is_product_change",
        ),
        pytest.param(
            "acceptance_pool_pos", 
            "acceptance_pool_pos_tso", 
            0.001, 
            0.001, 
            id="acceptance_pool_pos",
        ),
        pytest.param(
            "acceptance_pool_neg", 
            "acceptance_pool_neg_tso", 
            0.001, 
            0.001, 
            id="acceptance_pool_neg",
        ),
        pytest.param(
            "underfulfill_pool_pos", 
            "underfulfill_pool_pos_tso", 
            0.001, 
            0.001, 
            id="underfulfill_pool_pos",
        ),
        pytest.param(
            "underfulfill_pool_neg", 
            "underfulfill_pool_neg_tso", 
            0.001, 
            0.001, 
            id="underfulfill_pool_neg",
        ),
        pytest.param(
            "underfull_flag_pos", 
            "underfull_flag_pos_tso", 
            0, 
            0, 
            id="underfull_flag_pos",
        ),
        pytest.param(
            "underfull_flag_neg", 
            "underfull_flag_neg_tso", 
            0, 
            0, 
            id="underfull_flag_neg",
        ),
        pytest.param(
            "setpoint_pos", 
            "setpoint_pos_tso", 
            0.001, 
            0.001, 
            id="setpoint_pos",
        ),
        pytest.param(
            "setpoint_neg", 
            "setpoint_neg_tso", 
            0.001, 
            0.001, 
            id="setpoint_neg",
        ),
        pytest.param(
            "account_pos", 
            "account_pos_tso", 
            0.001, 
            0.001, 
            id="account_pos",
        ),
        pytest.param(
            "account_neg", 
            "account_neg_tso", 
            0.001, 
            0.001, 
            id="account_neg",
        ),
        pytest.param(
            "allocable_acceptance_pos", 
            "allocable_acceptance_pos_tso", 
            0.001, 
            0.001, 
            id="allocable_acceptance_pos",
        ),
        pytest.param(
            "allocable_acceptance_neg", 
            "allocable_acceptance_neg_tso", 
            0.001, 
            0.001, 
            id="allocable_acceptance_neg",
        ),
        pytest.param(
            "allocable_underfulfill_pos", 
            "allocable_underfulfill_pos_tso", 
            0.001, 
            0.001, 
            id="allocable_underfulfill_pos",
        ),
        pytest.param(
            "allocable_underfulfill_neg", 
            "allocable_underfulfill_neg_tso", 
            0.001, 
            0.001, 
            id="allocable_underfulfill_neg",
        ),
    ],
)
def test_columns(load_data, s1, s2, rtol, atol):
    tso_data=load_data
    for h in range(2, 25):
        df = tso_data[h]
        pd.testing.assert_series_equal(df[s1], df[s2], rtol=rtol, atol=atol, check_names=False)

    
