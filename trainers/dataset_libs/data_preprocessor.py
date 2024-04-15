# !/usr/bin/python3.8

"""
@ModuleName: dataset processing module to get target samples
@author: zhangmeixian
"""
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from configs.constants import column_time, column_label, label_normal, status_exception, status_exception_no, \
    column_data, column_source_path, column_time_range, column_exception_range, time_format_input_time
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
        self.second_interval = None


class DataPreprocessor(object):
    """
    dataset processing entrence
    """

    @classmethod
    def execute(cls, args):
        """
        excutor of dataset processing
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
            # if dataset_name == "SMD":
            #     cur_samples = DataPreprocessor.get_SMD_dataset(
            #         sample_time_window_before, sample_time_window_after, sample_day_window, pred_len)
            # elif dataset_name == "SMAP" or dataset_name == "MSL":
            #     cur_samples = DataPreprocessor.get_SMAP_and_MSL_dataset(
            #         sample_time_window_before, sample_time_window_after, sample_day_window, pred_len, dataset_name)
            if dataset_name == "NAB":
                cur_samples = DataPreprocessor.get_NAB_dataset(
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
    def get_NAB_dataset(cls, sample_time_window_before, sample_time_window_after, sample_day_window, pred_len):
        """
        get NAB dataset
        """
        data_root_path = "./data/origin_data/NAB/data"
        label_path = "./data/origin_data/NAB/labels/combined_windows.json"
        with open(label_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        total_samples, exception_count, normal_count = [], 0, 0
        for index_suffix, exception_windows in data.items():
            # if exception_windows is None or len(exception_windows) <= 0:
            #     continue
            exception_windows = [[x[0][:19], x[1][:19]] for x in exception_windows]
            exception_times = Timer.get_target_times_by_time_range(minute_interval=1, time_range=exception_windows)
            data_path = "/".join([data_root_path, index_suffix])
            if not os.path.exists(data_path):
                continue
            cur_source = pd.read_csv(data_path, index_col=0)
            cur_source = cur_source.reset_index().rename(columns={"timestamp": column_time, "value": column_data})
            cur_source[column_time] = cur_source[column_time].apply(lambda x: Timer.format_time_to_minute(x))
            cur_source.drop_duplicates(inplace=True)
            cur_source = DataPreprocessor.source_interpolation(cur_source, minute_interval=1)
            minute_interval = int(Timer.get_second_gap_between_time(
                cur_source[column_time].iloc[0], cur_source[column_time].iloc[1]) // 60)
            print("{}, minute interval: {}".format(index_suffix, minute_interval))
            cur_source[column_label] = cur_source[column_time].apply(
                lambda x: status_exception if x in exception_times else status_exception_no)
            cur_samples = DataPreprocessor.get_sample_objs(
                cur_source, "NAB", index_suffix, sample_time_window_before, sample_time_window_after, sample_day_window,
                pred_len)
            if cur_samples is None or len(cur_samples) <= 0:
                continue
            exception_count += len(cur_samples[0])
            normal_count += len(cur_samples[1])
            # if exception_count + normal_count >= 16078:
            #     break
            total_samples.append(cur_samples)
        sample_path = "./data/dataset/{}/{}.pickle".format("NAB", "NAB")
        Variabler.save_variable(total_samples, sample_path)
        logger.info("success to save {} samples: {}, total exception: {}, normal: {}".format(
            "NAB", sample_path, exception_count, normal_count))
        return total_samples

    @classmethod
    def get_CSM_dataset(cls, sample_time_window_before, sample_time_window_after, sample_day_window, pred_len):
        """
        get CSM dataset
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
            cur_source[column_time] = cur_source[column_time].apply(lambda x: Timer.format_time_to_minute(x))
            cur_source = DataPreprocessor.source_interpolation(cur_source, minute_interval=1)
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
        get AIOps dataset
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
            break
        sample_path = "./data/dataset/{}/{}.pickle".format("AIOps", "AIOps")
        Variabler.save_variable(total_samples, sample_path)
        logger.info("success to save {} samples: {}, total exception: {}, normal: {}".format(
            "AIOps", sample_path, exception_count, normal_count))
        return total_samples

    @classmethod
    def get_SkAB_dataset(cls, sample_time_window_before, sample_time_window_after, sample_day_window, pred_len):
        """
        get SkAB dataset
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
                source_df.drop(columns=["changepoint"], inplace=True)
                cur_samples = DataPreprocessor.get_sample_objs(
                    source_df, "SkAB", data_des, sample_time_window_before, sample_time_window_after,
                    sample_day_window, pred_len, second_interval=1)
                if cur_samples is None or len(cur_samples) <= 0:
                    continue
                exception_count += len(cur_samples[0])
                normal_count += len(cur_samples[1])
                total_samples.append(cur_samples)
        sample_path = "./data/dataset/{}/{}.pickle".format("SkAB", "SkAB")
        if not os.path.exists("./data/dataset/{}".format("SkAB")):
            os.makedirs("./data/dataset/{}".format("SkAB"))
        Variabler.save_variable(total_samples, sample_path)
        logger.info("success to save {} samples: {}, total exception: {}, normal: {}".format(
            "SkAB", sample_path, exception_count, normal_count))
        return total_samples

    @classmethod
    def get_SMAP_and_MSL_dataset(cls, sample_time_window_before, sample_time_window_after, sample_day_window, pred_len,
                                 dataset_name):
        """
        get SMAP & MSL dataset
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
        get anomaly index
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
        get WADI dataset
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
        test_new[column_time] = DataPreprocessor.get_time(test_new["Date"].tolist(), test_new["Time"].tolist())
        test_new.drop(columns=["Date", "Time", "Row"], inplace=True)
        sample_objs = DataPreprocessor.get_sample_objs(
            test_new, "WADI", "WADI_attackdataLABLE.csv", sample_time_window_before, sample_time_window_after,
            sample_day_window, pred_len, second_interval=1)
        sample_objs = [sample_objs]
        exception_count, normal_count = len(sample_objs[0][0]), len(sample_objs[0][1])
        sample_path = "./data/dataset/WADI/WADI.pickle"
        Variabler.save_variable(sample_objs, sample_path)
        logger.info("success to save WADI samples: {}, total exception: {}, normal: {}".format(
            sample_path, exception_count, normal_count))
        return sample_objs

    @classmethod
    def get_time(cls, date_strs, time_strs):
        """
        parse time for WADI dataset
        """
        time_list = []
        hour_count, data_count = 18, 0
        for date_str, time_str in zip(date_strs, time_strs):
            time_str = DataPreprocessor.convert_time_str(time_str, hour_count)
            raw_str = date_str + " " + time_str
            raw_str_datetime = datetime.strptime(raw_str, "%m/%d/%y %H:%M:%S")
            raw_time = raw_str_datetime.strftime("%Y-%m-%d %H:%M:%S")
            data_count += 1
            if data_count >= 3600 and data_count % 3600 == 0:
                hour_count += 1
            if hour_count >= 24:
                hour_count = 0
            time_list.append(raw_time)
        return time_list

    @classmethod
    def convert_time_str(cls, time_str, hour_count):
        """
        convert time string by adding hour info
        """
        # parse time
        minutes, seconds = int(time_str.split(':')[0]), int(time_str.split(':')[1][:2])
        time_str_modified = f"{hour_count:02d}:{minutes:02d}:{seconds:02d}"
        return time_str_modified

    @classmethod
    def get_SMD_dataset(cls, sample_time_window_before, sample_time_window_after, sample_day_window, pred_len):
        """
        get SMD dataset
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
                        sample_day_window, pred_len, minute_interval=1, sample_times=None, second_interval=None,
                        sample_limit_count=100000):
        """
        get sample objects
        """
        if source is None or len(source) <= 0:
            return None
        exception_samples, normal_samples = [], []
        count_gap = second_interval if second_interval is not None else minute_interval
        history_day_count = 0 if sample_day_window <= 0 else sample_day_window
        legal_source_count = (sample_time_window_before // count_gap + 1) * (sample_day_window + 1) + \
                             (sample_time_window_after // count_gap) * sample_day_window
        if sample_times is None:
            total_times = source[column_time].tolist()
            minute_interval = second_interval / 60 if second_interval is not None else minute_interval
            sample_start_index = (int(1440 / minute_interval)) * history_day_count + \
                                 sample_time_window_before * count_gap
            sample_times = total_times[sample_start_index:]
        sample_count = 0
        for sample_time in sample_times:
            if second_interval is not None:
                sample_time_list = Timer.get_dynamic_sample_second_time_list(
                    sample_time, second_interval, sample_day_window, sample_time_window_before,
                    sample_time_window_after, pred_len)
            else:
                sample_time_list = Timer.get_dynamic_sample_time_list(
                    sample_time, minute_interval, sample_day_window, sample_time_window_before,
                    sample_time_window_after, pred_len)
            sample_data = source[source[column_time].isin(sample_time_list)]
            sample_data = DataPreprocessor.fill_source_with_sample_time_list(sample_data, sample_time_list,
                                                                             is_log=False)
            if "datetime" in sample_data.columns:
                sample_data.drop(columns=["datetime"], inplace=True)
            sample_data.sort_values(by=column_time, inplace=True)
            if sample_data is None or len(sample_data) != legal_source_count:
                continue
            sample_obj = SampleObj()
            sample_obj.sample_time = sample_time
            sample_obj.sample_data = sample_data
            sample_obj.dataset = dataset
            sample_obj.data_des = data_des
            sample_obj.second_interval = second_interval if second_interval is not None else minute_interval * 60
            sample_obj.sample_label = source[source[column_time] == sample_time][column_label].iloc[0] if len(
                source[source[column_time] == sample_time]) > 0 else None
            sample_obj.unique_key = "_".join([dataset, str(data_des), sample_time])
            if sample_obj.sample_label is None:
                print("[{}-{}] no label sample: {}".format(dataset, data_des, sample_time))
                continue
            elif sample_obj.sample_label == status_exception:
                exception_samples.append(sample_obj)
            else:
                normal_samples.append(sample_obj)
            sample_count += 1
            print("\rsample obj process pace ratio: {}%".format(round((sample_count) / len(sample_times), 6) * 100),
                  end="")
        print("")
        logger.info("success to get samples for {}-{}, total exception samples: {}, total normal samples: {}".format(
            dataset, data_des, len(exception_samples), len(normal_samples)))
        return [exception_samples, normal_samples]

    @classmethod
    def source_interpolation(cls, source, minute_interval=1):
        """
        data interpolation
        """
        if source is None or len(source) <= 0:
            return source
        source_interpolation = source.copy()
        start_time, end_time = source_interpolation[column_time].min(), \
                               source_interpolation[column_time].max()
        target_time_list = Timer.get_target_time_list(
            minute_interval=minute_interval, start_time=start_time, end_time=end_time)
        exist_time_list = source_interpolation[column_time].tolist()
        loss_time_list = list(set(target_time_list).difference(set(exist_time_list)))
        if loss_time_list is None or len(loss_time_list) <= 0:
            return source_interpolation
        source_interpolation["datetime"] = source_interpolation[column_time].tolist()
        source_interpolation["datetime"] = pd.to_datetime(source_interpolation["datetime"])
        source_interpolation.index = source_interpolation["datetime"]
        source_interpolated = source_interpolation.resample("{}T".format(minute_interval)).asfreq().interpolate(
            method="linear")
        source_interpolated.drop(columns=[column_time, "datetime"], inplace=True)
        source_interpolated[column_time] = source_interpolated.index
        source_interpolated[column_time] = source_interpolated[column_time].apply(
            lambda x: x.strftime(time_format_input_time))
        source_interpolated.reset_index(drop=True, inplace=True)
        return source_interpolated

    @classmethod
    def fill_source_with_sample_time_list(cls, source, sample_time_list, is_log=True):
        """
        fill nan
        """
        if source is None or len(source) <= 0:
            return source
        if "KPI ID" in source:
            source.drop(columns="KPI ID", inplace=True)
        exist_time_list = source[column_time].tolist()
        loss_time_list = list(set(sample_time_list).difference(set(exist_time_list)))
        if loss_time_list is None or len(loss_time_list) <= 0 or len(loss_time_list) >= 0.4 * len(source):
            if is_log:
                # 数据缺失过多或者无缺失都不进行后续处理
                logger.info("no loss or too much loss time, loss details：{}-{}.".format(
                    len(loss_time_list) if loss_time_list is not None else 0, loss_time_list))
            time_label_cols = [column_time, column_label]
            data_cols = source.columns.difference(time_label_cols)
            column_means = source[data_cols].mean()
            df_filled = source.copy()
            df_filled[data_cols] = df_filled[data_cols].fillna(column_means)
            source.fillna(method="ffill", inplace=True)
            return source
        loss_source_df = pd.DataFrame(np.full((len(loss_time_list), len(source.columns)), np.nan),
                                      columns=source.columns)
        loss_source_df[column_time] = loss_time_list
        source = pd.concat([source, loss_source_df])
        filled_source = DataPreprocessor.fill_loss_by_median(source.copy())
        time_label_cols = [column_time, column_label]
        data_cols = filled_source.columns.difference(time_label_cols)
        column_means = filled_source[data_cols].mean()
        df_filled = filled_source.copy()
        df_filled[data_cols] = df_filled[data_cols].fillna(column_means)
        filled_source.fillna(method="ffill", inplace=True)
        return df_filled

    @classmethod
    def fill_loss_by_median(cls, source):
        """
        fill nan by median
        """
        try:
            if source is None or len(source) <= 0:
                return source
            if not source.isna().any().any():
                # 若无缺失值，则返回
                return source
            source["datetime"] = source[column_time].tolist()
            source["datetime"] = pd.to_datetime(source["datetime"])
            source.index = source["datetime"]
            # source_median = source.groupby([source.index.time]).transform("median")
            source_median = source.groupby(source.index.time).apply(DataPreprocessor.compute_median)
            filled_source = source.fillna(source_median)
            filled_source.reset_index(drop=True, inplace=True)
            filled_source.drop(columns=["datetime"], inplace=True)
            # 采用线性的方式进行无对应时间点位的缺失值填充
            filled_source.interpolate(inplace=True)
            return filled_source
        except Exception as e:
            return source

    @classmethod
    def compute_median(cls, sub_df):
        """
        get median
        """
        if sub_df is None or len(sub_df) <= 0:
            return None
        return sub_df.median()

    @classmethod
    def get_anomaly_details(cls, file_path):
        """
        get anomaly details
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
    def split_samples_by_class(cls, total_samples, train_ratio, test_ratio):
        """
        split samples by class (only train with normal samples, exceptions in test samples)
        """
        if total_samples is None or len(total_samples) <= 0:
            return None, None, None
        train_samples, valid_samples, test_samples = [], [], []
        for exception_samples, normal_samples in total_samples:
            if exception_samples is not None and len(exception_samples) > 0:
                test_samples.extend(exception_samples)
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
    def grousampling(cls, total_samples, train_ratio, test_ratio, dataset_name):
        """
        Grousampling
        """
        if total_samples is None or len(total_samples) <= 0:
            return None, None, None, None
        train_samples, valid_samples, test_samples, showed_samples = [], [], [], []
        for exception_samples, normal_samples in total_samples:
            if exception_samples is not None and len(exception_samples) > 0 and normal_samples is not None and len(
                    normal_samples) > 0 and len(showed_samples) <= 0 and dataset_name not in ["WADI"]:
                showed_samples.extend(exception_samples)
                showed_samples.extend(normal_samples)
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
        return train_samples, valid_samples, test_samples, showed_samples

    @classmethod
    def process_sample_data(cls, sample_data, sample_window_before, sample_window_after, sample_day_window):
        """
        process sample data
        """
        if sample_data is None or len(sample_data) <= 0:
            return None
        if column_label in sample_data:
            sample_data.drop(columns=column_label, inplace=True)
        if "KPI ID" in sample_data:
            sample_data.drop(columns="KPI ID", inplace=True)
        # print(list(sample_data.columns))
        # process history special outliers
        for column in sample_data.columns:
            if column == column_time:
                continue
            sample_data = DataPreprocessor.remove_history_outlier(
                sample_data, column, sample_window_before, sample_window_after, sample_day_window,
                anomaly_level_index=5)
        sample_data.sort_values(by=column_time, inplace=True)
        return sample_data

    @classmethod
    def remove_history_outlier(cls, source_df, column_name, sample_window_before, sample_window_after,
                               sample_day_window, anomaly_level_index=7):
        """
        remove extraordinary exception points in history data, substitute it by median value
        """
        if sample_day_window <= 0:
            # no history source to get
            return source_df
        try:
            if source_df is None or len(source_df) <= 0:
                return source_df
            history_source = source_df[:(sample_window_before + 1 + sample_window_after) * sample_day_window]
            cur_source = source_df[(sample_window_before + 1 + sample_window_after) * sample_day_window:]
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
