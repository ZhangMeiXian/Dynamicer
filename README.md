# TSDynamicer
Time Series Dynamic Anomaly Detection with Transformer

# Dependencies
python >= 3.6
> pip3 install -r requirements.txt

# Data Preparation

## single index dataset

AIOps challenge 2018: download and unzip data.
> [AIOps](https://smileyan.lanzoul.com/ixpcU03lp97g)

> [AIOps-github](https://github.com/NetManAIOps/KPI-Anomaly-Detection)

CSM: our non-public dataset CSM(Custom Server Metrics).

## multi index dataset
SMD: git download.
> [SMD](https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset)

SWaT&WADI(SWaT not used): apply for the data access permission.
> [application_url](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)

MSL&SMAP: terminate download and unzip data.
```shell
wget https://s3-us-west-2.amazonaws.com/telemanom/data.zip && unzip data.zip && rm data.zip
cd data && wget https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv
```

SkAB: git download.
> [SkAB](https://github.com/waico/SkAB)

## other unused dataset:

NAB: git download.
> [NAB](https://github.com/numenta/NAB)

# Data Preprocessing
all datasets will be processed to dataframe with the same sample_obj list: [sample_obj1, sample_obj2, ...],details:
> sample_obj.sample_time: current time string

> sample_obj.dataset: dataset name

> sample_obj.data_des: which column or file is used to get data

> sample_obj.sample_data: processed target data of current sample, processed sample df

> sample_obj.label: label of current sample, 0 for exception and 1 for normal

process script, e.g. SMD:
> python3 data_preprocessing.py --dataset SMD

# Get Dynamic Forecasting Started
> bash ./scripts/SMD/SMD_forecasting.sh

# Get Dynamic Anomaly Detection Started
> bash ./scripts/SMD/SMD_anomaly_detection.sh

# Results

# Citations

# Contact

# Acknowledgement

