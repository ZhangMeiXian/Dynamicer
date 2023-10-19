# TSDynamicer
Time Series Dynamic Anomaly Detection with Transformer

# Dependencies
python >= 3.6
> pip3 install -r requirements.txt

# Data Preparation
SMD: git download.
> [SMD](https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset)

SWat&WADI: apply for the data access permission.
> [SWat&WADI](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)

MSL&SMAP: terminate download and unzip data.
```shell
wget https://s3-us-west-2.amazonaws.com/telemanom/data.zip && unzip data.zip && rm data.zip
cd data && wget https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv
```

# Data Preprocessing
> python3 data_preprocessing.py --dataset_name SMD

# Get Dynamic Forecasting Started
> bash ./scripts/SMD/SMD_forecasting.sh

# Get Dynamic Anomaly Detection Started
> bash ./scripts/SMD/SMD_anomaly_detection.sh

# Results

# Citations

# Contact

# Acknowledgement

