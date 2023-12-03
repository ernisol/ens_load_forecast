# Installation

Running the following commands on linux will download and install the module:

```bash
git clone https://github.com/ernisol/ens_load_forecast.git
python3 -m venv venv
source venv/bin/activate
pip install -U pip wheel setuptools
pip install .
```

Then put the data (csv files) `ens_load_forecast/data`.

Finally, run the preprocessing and modelling with the following command:

```bash
python -m ens_load_forecast
```

Once pre-processing and modelling is done (allow up to 5 minutes), use a notebook to explore the data and model results.
