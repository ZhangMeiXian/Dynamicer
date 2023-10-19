# !/usr/bin/python3.8

"""
@ModuleName: 数据集预处理模块
@author: zhangmeixian
"""
import os
import numpy as np
import pandas as pd
from configs.constants import column_time, column_label


class DataPreprocessor(object):
    """
    数据集合处理入口
    """

    @classmethod
    def execute(cls, args):
        """
        执行数据处理
        """
        if args is None or args.dataset is None:
            print("illegal input, please check")
            return
        dataset_list = args.dataset
        for dataset_name in dataset_list:
            if dataset_name == "SMD":
                DataPreprocessor.get_SMD_dataset()
            elif dataset_name == "SMAP" or dataset_name == "MSL":
                pass
            elif dataset_name == "WADI":
                pass
            elif dataset_name == "SWaT":
                pass
            elif dataset_name == "custom":
                # custom dataset can be added here
                pass
            else:
                continue

    @classmethod
    def get_SMD_dataset(cls):
        """
        处理并获取SMD数据集
        """
        root_path = "./data/origin_data/OmniAnomaly/ServerMachineDataset"
        data_saved_path = "./data/dataset/SMD"
        if not os.path.exists(data_saved_path):
            os.makedirs(data_saved_path)
        # supervised method can only use labeled data
        data_dir = os.path.join(root_path, "test")
        label_dir = os.path.join(root_path, "test_label")
        anomaly_detail_dir = os.path.join(root_path, "interpretation_label")
        for filename in os.listdir(data_dir):
            if filename.endswith(".txt"):
                data_file = os.path.join(data_dir, filename)
                label_file = os.path.join(label_dir, filename)
                anomaly_detail = os.path.join(anomaly_detail_dir, filename)
                data_tmp = np.genfromtxt(data_file, dtype=float, delimiter=",")
                if data_tmp is None or len(data_tmp) <= 0:
                    continue
                label_tmp = np.genfromtxt(label_file, dtype=int, delimiter=",")
                anomaly_details = DataPreprocessor.get_anomaly_details(anomaly_detail)
                column_list = []
                for i in range(len(data_tmp[0, :])):
                    column_list.append("index_{}".format(i + 1))
                cur_source = pd.DataFrame(data_tmp, columns=column_list).reset_index()
                cur_source.rename(columns={"index": column_time}, inplace=True)
                cur_source[column_label] = label_tmp.tolist()
                print()

    @classmethod
    def get_anomaly_details(cls, file_path):
        """
        获取异常指标详情
        """
        anomaly_details = {}
        with open(file_path, "r") as f:
            for line in f:
                key, value = line.strip().split(":")
                value_list = list(map(int, value.split(",")))







