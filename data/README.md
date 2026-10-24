# data

nothing is stored here for the default dataset. mauna loa co2 ships with
statsmodels and is loaded in `src/data.py`.

if a dataset has to be downloaded (e.g. an irish cso or eirgrid series), drop the
raw file in this folder and add a loader for it in `src/data.py`.
