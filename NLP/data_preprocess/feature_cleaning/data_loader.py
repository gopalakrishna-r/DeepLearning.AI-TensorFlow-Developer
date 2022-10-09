import os
import tarfile
from urllib import request
import pathlib
import pandas as pd
DOWNLOAD_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = pathlib.Path("datasets", "housing")
HOUSING_URL = DOWNLOAD_URL + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    housing_path.mkdir(parents=True, exist_ok=True)
    tgz_path = pathlib.Path.joinpath(housing_path, "housing.tgz")
    request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    fetch_housing_data()
    csv_path = pathlib.Path.joinpath(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


