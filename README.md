<a href="https://www.e2m.energy/"><img src="https://user-images.githubusercontent.com/8255114/148765040-975650b6-1db2-4537-aac4-0840f28bf678.png" alt="e2m logo" title="e2m" height="50" align="right"></a>

# aFRR remuneration

A tool to calculate the aFRR remuneration for the european energy market.

## About

This project was initiated with the start of aFRR remuneration in temporal resolution of seconds on October 1st 2021 
which is one further step to fulfill the EU target market design.
The motivation for creating this python package is to provide a tool for evaluating remuneration of aFRR activation 
events by TSOs.
Therefore, it provides an implementation of the calculation procedure described in the 
[model description](https://www.regelleistung.net/ext/download/Modellbeschreibung_aFRR-Abrechnung_ab_01.10.2021) as 
python code.


## Installation 

Install the latest version available on [pypi.org](https://pypi.org/project/afrr-remuneration/)

```bash
pip install afrr-remuneration
```

If you are looking for a development installation, read [here](#development-installation) about how to install the package from sources.

## Usage

Here is some example code that shows how use functionality of this package. 
Make sure you have a file at hand with data about setpoints and actual values of an aFRR activation event. See the 
example files from 
[regelleistung.net](https://www.regelleistung.net/ext/download/Beispieldateien_aFRR-Abrechnung_ab_01.10.2021) to 
understand the required file format.
Note, you have to make sure that data starts at the beginning of an aFRR activation event.

````python 
from afrr_remuneration.aFRR import calc_acceptance_tolerance_band, calc_underfulfillment_and_account
from afrr_remuneration.data import parse_tso_data

# load the setpoint and the measured value for example by reading the tso data
file = "20211231_aFRR_XXXXXXXXXXXXXXXX_XXX_PT1S_043_V01.csv"
tso_df = parse_tso_data(file)[0]

# calculate the tolerance band 
band_df = calc_acceptance_tolerance_band(
    setpoint=tso_df["setpoint"], measured=tso_df["measured"]
    )

# calculate acceptance values and other relevant series like the under-/overfulfillment 
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

We plan to

- [ ] Add a testfile with artificial data
- [ ] Add an example with a valid MOL

Feel free to help us here!

## Contributing

Contributions are highly welcome. For more details, please have a look in to 
[contribution guidelines](https://github.com/energy2market/afrr-remuneration/blob/main/CONTRIBUTING.md).

### Development installation

For installing the package from sources, please clone the repository with

```bash
git clone git@github.com:energy2market/afrr-remuneration.git
```

Then, in the directory `afrr-remuneration` (the one the source code was cloned to), execute

```bash
poetry install
```

which creates a virtual environment under `./venv` and installs required package and the package itself to this virtual environment.
Read here for more information about <a href="https://python-poetry.org/">poetry</a>.
