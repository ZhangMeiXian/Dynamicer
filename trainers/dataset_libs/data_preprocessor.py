# !/usr/bin/python3.8

"""
@ModuleName: 数据集预处理模块
@author: zhangmeixian
"""
import json
import os
import numpy as np
import pandas as pd
from configs.constants import column_time, column_label, label_normal, status_exception, status_exception_no, \
    column_data, column_source_path, column_time_range, column_exception_range
from data.origin_data.CSM.metric_configs import DATA_DETAILS
from commons.timer import Timer
from commons.variabler import Variabler
from commons.logger import get_logger

logger = get_logger(__name__)


class SampleObj(object):
    """
    样本数据集对象
    """

    def __init__(self):
        self.dataset = None
        self.data_des = None
        self.sample_data = None
        self.sample_label = None
        self.sample_time = None


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
        dataset_list = args.dataset if isinstance(args.dataset, list) else [args.dataset]
        sample_time_window_before = args.sample_time_window_before
        sample_time_window_after = args.sample_time_window_after
        sample_day_window = args.sample_day_window
        pred_len = args.pred_len
        total_samples = []
        for dataset_name in dataset_list:
            if dataset_name == "SMD":
                cur_samples = DataPreprocessor.get_SMD_dataset(
                    sample_time_window_before, sample_time_window_after, sample_day_window, pred_len)
            elif dataset_name == "SMAP" or dataset_name == "MSL":
                cur_samples = DataPreprocessor.get_SMAP_and_MSL_dataset(
                    sample_time_window_before, sample_time_window_after, sample_day_window, pred_len, dataset_name)
            elif dataset_name == "WADI":
                cur_samples = DataPreprocessor.get_WADI_dataset(
                    sample_time_window_before, sample_time_window_after, sample_day_window, pred_len)
            elif dataset_name == "SkAB":
                cur_samples = DataPreprocessor.get_SkAB_dataset(
                    sample_time_window_before, sample_time_window_after, sample_day_window, pred_len)
            elif dataset_name == "AIOps":
                cur_samples = DataPreprocessor.get_AIOps_dataset(
                    sample_time_window_before, sample_time_window_after, sample_day_window, pred_len)
            elif dataset_name == "CSM":
                # our custom dataset: Custom Server Metrics
                cur_samples = DataPreprocessor.get_CSM_dataset(
                    sample_time_window_before, sample_time_window_after, sample_day_window, pred_len)
            else:
                cur_samples = []
            if cur_samples is not None and len(cur_samples) > 0:
                total_samples.append(cur_samples)
        return total_samples

    @classmethod
    def get_CSM_dataset(cls, sample_time_window_before, sample_time_window_after, sample_day_window, pred_len):
        """
        获取自定义数据集
        """
        index_ids = list(DATA_DETAILS.keys())
        total_samples, exception_count, normal_count = [], 0, 0
        for index_id in index_ids:
            source_path = DATA_DETAILS[index_id][column_source_path]
            sample_time_range = DATA_DETAILS[index_id][column_time_range]
            sample_exception_range = DATA_DETAILS[index_id][column_exception_range]
            sample_times = Timer.get_target_times_by_time_range(1, sample_time_range)
            exception_times = Timer.get_target_times_by_time_range(1, sample_exception_range)
            cur_source = pd.read_csv(source_path, index_col=0)
            cur_source[column_label] = cur_source[column_time].apply(
                lambda x: status_exception if x in exception_times else status_exception_no)
            cur_samples = DataPreprocessor.get_sample_objs(
                cur_source, "CSM", index_id, sample_time_window_before, sample_time_window_after, sample_day_window,
                pred_len, sample_times=sample_times)
            if cur_samples is None or len(cur_samples) <= 0:
                continue
            exception_count += len(cur_samples[0])
            normal_count += len(cur_samples[1])
            total_samples.append(cur_samples)
        sample_path = "./data/dataset/{}/{}.pickle".format("CSM", "CSM")
        Variabler.save_variable(total_samples, sample_path)
        logger.info("success to save {} samples: {}, total exception: {}, normal: {}".format(
            "CSM", sample_path, exception_count, normal_count))
        return total_samples

    @classmethod
    def get_AIOps_dataset(cls, sample_time_window_before, sample_time_window_after, sample_day_window, pred_len):
        """
        获取并处理AIOps数据集
        """
        root_dir = "./data/origin_data/AIOps"
        ground_truth_file = os.path.join(root_dir, "phase2_ground_truth.hdf")
        data = pd.read_hdf(ground_truth_file)
        data.rename(columns={"timestamp": column_time, "value": column_data}, inplace=True)
        data[column_time] = data[column_time].apply(lambda x: Timer.timestamp_to_str(x))
        grouped_data = data.groupby("KPI ID")
        total_samples, exception_count, normal_count = [], 0, 0
        for kpi_id, kpi_source in grouped_data:
            if kpi_source is None or len(kpi_source) <= 0:
                continue
            cur_samples = DataPreprocessor.get_sample_objs(
                kpi_source, "AIOps", kpi_id, sample_time_window_before, sample_time_window_after, sample_day_window,
                pred_len)
            if cur_samples is None or len(cur_samples) <= 0:
                continue
            exception_count += len(cur_samples[0])
            normal_count += len(cur_samples[1])
            total_samples.append(cur_samples)
        sample_path = "./data/dataset/{}/{}.pickle".format("AIOps", "AIOps")
        Variabler.save_variable(total_samples, sample_path)
        logger.info("success to save {} samples: {}, total exception: {}, normal: {}".format(
            "AIOps", sample_path, exception_count, normal_count))
        return total_samples

    @classmethod
    def get_SkAB_dataset(cls, sample_time_window_before, sample_time_window_after, sample_day_window, pred_len):
        """
        获取并处理SkAB数据集
        """
        root_dir = "./data/origin_data/SkAB"
        data_dirs = ["data/valve1", "data/valve2", "data/other"]
        total_samples, exception_count, normal_count = [], 0, 0
        for data_dir in data_dirs:
            data_path = os.path.join(root_dir, data_dir)
            for filename in os.listdir(data_path):
                if not filename.endswith(".csv"):
                    continue
                data_des = "/".join([data_dir, filename])
                file_path = os.path.join(data_path, filename)
                source_df = pd.read_csv(file_path, sep=";")
                source_df.rename(columns={"anomaly": column_label, "datetime": column_time}, inplace=True)
                source_df[column_label] = source_df[column_label].apply(
                    lambda x: status_exception if x == 1 else status_exception_no)
                source_df[column_time] = Timer.generate_time_list(len(source_df))
                source_df.drop(columns=["changepoint"], inplace=True)
                cur_samples = DataPreprocessor.get_sample_objs(
                    source_df, "SkAB", data_des, sample_time_window_before, sample_time_window_after,
                    sample_day_window, pred_len)
                if cur_samples is None or len(cur_samples) <= 0:
                    continue
                exception_count += len(cur_samples[0])
                normal_count += len(cur_samples[1])
                total_samples.append(cur_samples)
        sample_path = "./data/dataset/{}/{}.pickle".format("SkAB", "SkAB")
        Variabler.save_variable(total_samples, sample_path)
        logger.info("success to save {} samples: {}, total exception: {}, normal: {}".format(
            "SkAB", sample_path, exception_count, normal_count))
        return total_samples

    @classmethod
    def get_SMAP_and_MSL_dataset(cls, sample_time_window_before, sample_time_window_after, sample_day_window, pred_len,
                                 dataset_name):
        """
        获取SMAP&MSL数据集
        """
        root_dir = "./data/origin_data/MSL&SMAP"
        test_dir = os.path.join(root_dir, "test")
        label_path = os.path.join(root_dir, "labeled_anomalies.csv")
        label_df = pd.read_csv(label_path)
        total_samples, exception_count, normal_count = [], 0, 0
        for filename in os.listdir(test_dir):
            if not filename.endswith(".npy"):
                continue
            chan_id = filename.split(".")[0]
            cur_label_df = label_df[label_df["chan_id"] == chan_id]
            if cur_label_df.empty:
                continue
            cur_dataset = cur_label_df["spacecraft"].iloc[0]
            if cur_dataset != dataset_name:
                continue
            anomaly_sequences = json.loads(cur_label_df["anomaly_sequences"].iloc[0])
            total_anomaly_index = DataPreprocessor.get_total_anomaly_index(anomaly_sequences)
            labels = []
            file_path = os.path.join(test_dir, filename)
            data_array = np.load(file_path)
            source_df = pd.DataFrame(data_array)
            source_df[column_time] = Timer.generate_time_list(len(source_df))
            for i in range(len(source_df)):
                if i in total_anomaly_index:
                    labels.append(status_exception)
                else:
                    labels.append(status_exception_no)
            source_df[column_label] = labels
            cur_samples = DataPreprocessor.get_sample_objs(
                source_df, dataset_name, filename, sample_time_window_before, sample_time_window_after,
                sample_day_window, pred_len)
            if cur_samples is None or len(cur_samples) <= 0:
                continue
            exception_count += len(cur_samples[0])
            normal_count += len(cur_samples[1])
            total_samples.append(cur_samples)
        sample_path = "./data/dataset/{}/{}.pickle".format(dataset_name, dataset_name)
        Variabler.save_variable(total_samples, sample_path)
        logger.info("success to save {} samples: {}, total exception: {}, normal: {}".format(
            dataset_name, sample_path, exception_count, normal_count))
        return total_samples

    @classmethod
    def get_total_anomaly_index(cls, anomaly_sequences):
        """
        获取所有的异常索引
        """
        if anomaly_sequences is None or len(anomaly_sequences) <= 0:
            return []
        total_anomaly_index = []
        for anomaly_sequence in anomaly_sequences:
            index_start, index_end = anomaly_sequence[0], anomaly_sequence[-1]
            for index in range(index_start, index_end + 1):
                total_anomaly_index.append(index)
        return list(set(total_anomaly_index))

    @classmethod
    def get_WADI_dataset(cls, sample_time_window_before, sample_time_window_after, sample_day_window, pred_len):
        """
        获取WADI数据集
        """
        test_new = pd.read_csv("./data/origin_data/WADI/WADI_attackdataLABLE.csv", skiprows=1)
        test_new.columns = test_new.columns.str.strip()
        # delete columns that total nan.
        ncolumns = ['2_LS_001_AL', '2_LS_002_AL', '2_P_001_STATUS', '2_P_002_STATUS']
        test_new.drop(columns=ncolumns, inplace=True)
        # transform label to target format.
        test_new.rename(columns={'Attack LABLE (1:No Attack, -1:Attack)': column_label}, inplace=True)
        test_new.loc[test_new['label'] == 1, 'label'] = status_exception_no
        test_new.loc[test_new['label'] == -1, 'label'] = status_exception
        test_new = test_new.fillna(method="ffill")
        # get time column
        test_new[column_time] = test_new.apply(lambda x: DataPreprocessor.get_time(x.Date, x.Time), axis=1)
        test_new.drop(columns=["Date", "Time", "Row"], inplace=True)
        sample_objs = DataPreprocessor.get_sample_objs(
            test_new, "WADI", "WADI_attackdataLABLE.csv", sample_time_window_before, sample_time_window_after,
            sample_day_window, pred_len)
        sample_objs = [sample_objs]
        exception_count, normal_count = len(sample_objs[0][0]), len(sample_objs[0][1])
        sample_path = "./data/dataset/WADI/WADI.pickle"
        Variabler.save_variable(sample_objs, sample_path)
        logger.info("success to save WADI samples: {}, total exception: {}, normal: {}".format(
            sample_path, exception_count, normal_count))
        return sample_objs

    @classmethod
    def get_time(cls, data_str, time_str):
        """
        获取时间
        """
        day, month, year = data_str.split("/")
        year = "20" + year
        month = "0" + month if len(month) == 1 else month
        time_str = time_str[:5] + ":00"
        final_time = year + "-" + month + "-" + day + " " + time_str
        return final_time

    @classmethod
    def get_SMD_dataset(cls, sample_time_window_before, sample_time_window_after, sample_day_window, pred_len):
        """
        处理并获取SMD数据集
        """
        root_path = "./data/origin_data/SMD/ServerMachineDataset"
        data_saved_path = "./data/dataset/SMD"
        if not os.path.exists(data_saved_path):
            os.makedirs(data_saved_path)
        # supervised method can only use labeled data
        data_dir = os.path.join(root_path, "test")
        label_dir = os.path.join(root_path, "test_label")
        # anomaly_detail_dir = os.path.join(root_path, "interpretation_label")
        total_samples, exception_count, normal_count = [], 0, 0
        for filename in os.listdir(data_dir):
            if not filename.endswith(".txt"):
                continue
            data_file = os.path.join(data_dir, filename)
            label_file = os.path.join(label_dir, filename)
            # anomaly_detail = os.path.join(anomaly_detail_dir, filename)
            data_tmp = np.genfromtxt(data_file, dtype=float, delimiter=",")
            if data_tmp is None or len(data_tmp) <= 0:
                continue
            data_label = np.genfromtxt(label_file, dtype=int, delimiter=",").tolist()
            data_label = [status_exception if x != label_normal else status_exception_no for x in data_label]
            # anomaly_details, total_anomaly_index_list = DataPreprocessor.get_anomaly_details(anomaly_detail)
            column_list = []
            for i in range(len(data_tmp[0, :])):
                column_list.append(i + 1)
            cur_source = pd.DataFrame(data_tmp, columns=column_list).reset_index()
            cur_source.rename(columns={"index": column_time}, inplace=True)
            cur_source[column_time] = Timer.generate_time_list(len(cur_source))
            cur_source[column_label] = data_label
            cur_samples = DataPreprocessor.get_sample_objs(
                cur_source, "SMD", filename, sample_time_window_before, sample_time_window_after, sample_day_window,
                pred_len)
            if cur_samples is None or len(cur_samples) <= 0:
                continue
            exception_count += len(cur_samples[0])
            normal_count += len(cur_samples[1])
            total_samples.append(cur_samples)
        saved_file_name = "./data/dataset/SMD/SMD.pickle"
        Variabler.save_variable(total_samples, saved_file_name)
        logger.info("success to save total SMD dataset: {}, total exception: {}, normal: {}".format(
            saved_file_name, exception_count, normal_count))
        return total_samples

    @classmethod
    def get_sample_objs(cls, source, dataset, data_des, sample_time_window_before, sample_time_window_after,
                        sample_day_window, pred_len, minute_interval=1, sample_times=None):
        """
        获取SMD样本数据集（数据均插值处理成1min级别的数据，然后再获取样本）
        """
        if source is None or len(source) <= 0:
            return None
        exception_samples, normal_samples = [], []
        legal_source_count = (sample_time_window_before // minute_interval + 1 +
                              sample_time_window_after // minute_interval) * (sample_day_window + 1)
        if sample_times is None:
            total_times = source[column_time].tolist()
            sample_start_index = (1440 // minute_interval) * (sample_day_window + 1) + \
                                 sample_time_window_before * minute_interval
            sample_times = total_times[sample_start_index:]
        for sample_time in sample_times:
            sample_time_list = Timer.get_dynamic_sample_time_list(
                sample_time, minute_interval, sample_day_window, sample_time_window_before, sample_time_window_after,
                pred_len)
            sample_data = source[source[column_time].isin(sample_time_list)]
            if sample_data is None or len(sample_data) != legal_source_count:
                continue
            sample_obj = SampleObj()
            sample_obj.sample_time = sample_time
            sample_obj.sample_data = sample_data
            sample_obj.dataset = dataset
            sample_obj.data_des = data_des
            sample_obj.sample_label = source[source[column_time] == sample_time][column_label].iloc[0]
            if sample_obj.sample_label == status_exception:
                exception_samples.append(sample_obj)
            else:
                normal_samples.append(sample_obj)
        logger.info("success to get samples for {}-{}, total exception samples: {}, total normal samples: {}".format(
            dataset, data_des, len(exception_samples), len(normal_samples)))
        return [exception_samples, normal_samples]

    @classmethod
    def get_anomaly_details(cls, file_path):
        """
        获取异常指标详情
        """
        anomaly_details, total_anomaly_index_list = {}, []
        with open(file_path, "r") as f:
            for line in f:
                key, value = line.strip().split(":")
                value_list = list(map(int, value.split(",")))
                total_anomaly_index_list.extend(value_list)
                key_range = key.split("-")
                key_start, key_end = int(key_range[0]), int(key_range[1])
                for ith in range(key_start, key_end + 1):
                    anomaly_details[ith] = value_list
        return anomaly_details, list(set(total_anomaly_index_list))

    @classmethod
    def split_samples(cls, total_samples, train_ratio, test_ratio):
        """
        将样本分层采样数据集划分程训练、验证和测试数据集
        """
        if total_samples is None or len(total_samples) <= 0:
            return None, None, None
        train_samples, valid_samples, test_samples = [], [], []
        for exception_samples, normal_samples in total_samples:
            if exception_samples is not None and len(exception_samples) > 0:
                index1, index2 = int(len(exception_samples) * train_ratio), int(
                    len(exception_samples) * (1 - test_ratio))
                train_samples.extend(exception_samples[:index1])
                if index2 < len(exception_samples):
                    valid_samples.extend(exception_samples[index1: index2])
                if index2 < len(exception_samples):
                    test_samples.extend(exception_samples[index2:])
            if normal_samples is not None and len(normal_samples) > 0:
                index1, index2 = int(len(normal_samples) * train_ratio), int(
                    len(normal_samples) * (1 - test_ratio))
                train_samples.extend(normal_samples[:index1])
                if index2 < len(normal_samples):
                    valid_samples.extend(normal_samples[index1: index2])
                if index2 < len(normal_samples):
                    test_samples.extend(normal_samples[index2:])
        return train_samples, valid_samples, test_samples

    @classmethod
    def process_sample_data(cls, sample_data, sample_window_before, sample_window_after, sample_day_window):
        """
        process sample data
        """
        if sample_data is None or len(sample_data) <= 0:
            return
        if column_label in sample_data:
            sample_data.drop(columns=column_label, inplace=True)
        # process history special outliers
        for column in sample_data.columns:
            if column == column_time:
                continue
            sample_data = DataPreprocessor.remove_history_outlier(
                sample_data, column, sample_window_before, sample_window_after, sample_day_window)
        sample_data.sort_values(by=column_time, inplace=True)
        return sample_data

    @classmethod
    def remove_history_outlier(cls, source_df, column_name, sample_window_before, sample_window_after,
                               sample_day_window, anomaly_level_index=7):
        """
        remove extraordinary exception points in history data, substitute it by median value
        """
        try:
            if source_df is None or len(source_df) <= 0:
                return source_df
            history_source = source_df[:(sample_window_before + 1 + sample_window_after) * (sample_day_window - 1)]
            cur_source = source_df[(sample_window_before + 1 + sample_window_after) * (sample_day_window - 1):]
            Q1 = DataPreprocessor.percentile(history_source[column_name].tolist(), 0.25)
            Q3 = DataPreprocessor.percentile(history_source[column_name].tolist(), 0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - anomaly_level_index * IQR
            upper_bound = Q3 + anomaly_level_index * IQR
            # 使用中位数替换异常值
            history_source.loc[
                (history_source[column_name] < lower_bound) | (history_source[column_name] > upper_bound),
                column_name] = history_source[column_name].median()
            processed_source = pd.concat([history_source, cur_source])
            return processed_source
        except Exception as e:
            logger.info("failed to remove outlier for: {}".format(str(e)))
            return source_df

    @classmethod
    def percentile(cls, data_list, percentile):
        """
        percentile calculation
        """
        try:
            if data_list is None or len(data_list) <= 0:
                return None
            data_sorted = sorted(data_list)
            n = len(data_sorted)
            k = (n - 1) * percentile
            f = int(k)
            c = k - f
            return (1 - c) * data_sorted[f] + c * data_sorted[f + 1]
        except Exception as e:
            logger.error("failed to get percentile, error: {}".format(str(e)))
            return None