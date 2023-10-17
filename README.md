# TSDynamicer
Time Series Dynamic Anomaly Detection with Transformer

# Dependencies
python >= 3.6
> pip3 install -r requirements.txt

# Data Preparation
SMT&SMAP: git download.
> [SMD](https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset)
> [SMAP](https://github.com/khundman/telemanom)

SWat&WADI: apply for the data access permission.
> [SWat](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)
> [WADI](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)

MSL&SMAP: terminate download and unzip data.
```shell
wget https://s3-us-west-2.amazonaws.com/telemanom/data.zip && unzip data.zip && rm data.zip
cd data && wget https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv
```

# Data Preprocessing

