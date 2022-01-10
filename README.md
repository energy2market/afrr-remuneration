<a href="https://www.e2m.energy/">
    <img src="https://www.e2m.energy/static/img/logos/e2m_logo.png" 
    alt="e2m logo" title="e2m" height="50" align="right" />
</a>

# aFRR remuneration

A tool to calculate the aFRR remuneration for the european energy market.

## About

This project is based on the 
<a href="https://www.regelleistung.net/ext/download/Modellbeschreibung_aFRR-Abrechnung_ab_01.10.2021">
    model description (in german from regelleistung.net)
</a> 
for the aFRR remuneration to fulfill the EU target market design.

## Installation 

For now the only way to install the tool is by cloning this repository. We aim to release a package on PyPi.
All dependencies can be installed using 
<a href="https://python-poetry.org/">poetry</a>. 

## Usage


````python 
from afrr_renumeration.aFRR import calc_acceptance_tolerance_band, calc_underfulfillment_and_account
from afrr_renumeration.data import parse_tso_data

# load the setpoint and the measured value for example by reading the tso data
file = "20211231_aFRR_XXXXXXXXXXXXXXXX_XXX_PT1S_043_V01.csv"
tso_df = parse_tso_data(file)

# calculate the tolerance band 
band_df = calc_acceptance_tolerance_band(
    setpoint=tso_df["setpoint"], measured=tso_df["measured"]
    )

# calculate acceptance values and other relevant serieses like the under-/overfulfillment 
underful_df = calc_underfulfillment_and_account(
    setpoint=band_df.setpoint,
    measured=band_df.measured,
    upper_acceptance_limit=band_df.upper_acceptance_limit,
    lower_acceptance_limit=band_df.lower_acceptance_limit,
    lower_tolerance_limit=band_df.lower_tolerance_limit,
    upper_tolerance_limit=band_df.upper_tolerance_limit,
)


````

## Next Steps

- [ ] Add a testfile with artificial data
- [ ] Add an example with a valid MOL

## Contributing

Contributions are highly welcome. For more details, please have a look in to 
[contribution guidelines](https://github.com/energy2market/afrr-remuneration/blob/main/CONTRIBUTING.md).
