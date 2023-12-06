# !/usr/bin/python3.8

"""
@ModuleName: module to process and get dataset
@author: zhangmeixian
"""
import numpy as np
import pandas as pd
import warnings
import chinese_calendar
from datetime import date, datetime, timedelta
from workalendar.asia import Singapore, China
from workalendar.usa.core import UnitedStates
from workalendar.europe import Russia
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from trainers.dataset_libs.data_preprocessor import DataPreprocessor
from trainers.methods.net_libs.timefeatures import time_features
from configs.constants import column_time, status_exception, time_format_input_time, complete_weight, weak_weight, \
    zero_weight
from sklearn.decomposition import PCA
from commons.logger import get_logger
from commons.timer import Timer

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

# config here with your own country
CALENDARS = {
    "China": chinese_calendar,
    "Russia": Russia(),
    "Singapore": Singapore(),
    "UnitedStates": UnitedStates()
}


class DatasetGenerator(Dataset):
    """
    dataset process and generate
    """

    def __init__(self, forcast_task="S", target_index="data", scale=True, timeenc=0, freq='h', samples=None,
                 sample_time_window_before=30, sample_time_window_after=0, sample_day_window=14, flag=None,
                 region=None, neighbor_window=None, is_reduce_dim=None, target_dim=None, seq_len=None):
        # basic info
        self.forcast_task = forcast_task
        self.target_index = target_index
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.samples = samples
        self.sample_time_window_before = sample_time_window_before
        self.sample_time_window_after = sample_time_window_after
        self.sample_day_window = sample_day_window
        self.flag = flag
        self.region = region
        self.seq_len = seq_len
        self.neighbor_window = neighbor_window
        self.neighbor_window = self.neighbor_window \
            if self.neighbor_window <= self.sample_time_window_before else self.sample_time_window_before
        self.is_reduce_dim = is_reduce_dim
        self.target_dim = target_dim
        # get the processed dataset
        self.__read_data__()
        logger.info("success to init dataset")

    def __read_data__(self):
        """
        get dataset
        """
        self.scaler = StandardScaler()
        data_x, data_y, data_x_marks, data_y_marks, data_sources, data_labels = [], [], [], [], [], []
        data_special_day_weights, data_neighbor_time_weights, data_unique_keys = [], [], []
        # get the data
        for i in range(len(self.samples)):
            sample_obj = self.samples[i]
            sample_label = sample_obj.sample_label
            sample_data = sample_obj.sample_data
            second_interval = sample_obj.second_interval
            sample_unique_key = sample_obj.unique_key
            sample_data.columns = [str(x) for x in sample_data.columns]
            if sample_data is None or len(sample_data) <= 0:
                continue
            # data preprocess, filter extraordinary exception data and substitute by median
            sample_data = DataPreprocessor.process_sample_data(
                sample_data, self.sample_time_window_before, self.sample_time_window_after, self.sample_day_window)
            if self.is_reduce_dim:
                pca = PCA(n_components=self.target_dim)
                feature_columns = [x for x in list(sample_data.columns) if x != column_time]
                features_only = sample_data[feature_columns]
                new_features = pca.fit_transform(features_only)
                sample_data_new = pd.DataFrame(new_features)
                sample_data_new[column_time] = sample_data[column_time].tolist()
                sample_data = sample_data_new
            scalar_index = (self.sample_time_window_before + 1 + self.sample_time_window_after) * \
                          self.sample_day_window \
                if self.sample_day_window > 0 else int(self.sample_time_window_before * 2 / 3)
            if self.forcast_task == 'M' or self.forcast_task == 'MS':
                cols_data = [col for col in sample_data.columns if col != column_time]
                origin_history_target = sample_data[:scalar_index][cols_data]
            else:
                origin_history_target = sample_data[:scalar_index][[self.target_index]]
            self.scaler.fit(origin_history_target.values)
            target_data, target_marks = self.get_target_data_and_time_vector(sample_data)
            history_data_count = (self.sample_time_window_before + 1 + self.sample_time_window_after) * \
                                 self.sample_day_window + self.sample_time_window_before
            history_data = target_data[:history_data_count]
            history_marks = target_marks[:history_data_count]
            if sample_label == status_exception:
                # get normal predict adjusted value
                end_index = -self.sample_time_window_before if self.sample_day_window > 0 else -1
                start_index = 0 if self.sample_day_window > 0 else -int(self.sample_time_window_before * 1/3)
                history_df = pd.DataFrame(history_data[start_index:end_index])
                medians = history_df.median()
                pred_data = pd.DataFrame([medians]).values
            else:
                pred_data = target_data[history_data_count:]
            pred_marks = target_marks[history_data_count:]
            data_x.append(history_data)
            data_x_marks.append(history_marks)
            data_y.append(pred_data)
            data_y_marks.append(pred_marks)
            if self.forcast_task == 'M' or self.forcast_task == 'MS':
                source_col = [x for x in sample_data.columns if x != column_time]
                data_sources.append(sample_data[source_col].values)
            else:
                data_sources.append(sample_data[[self.target_index]].values)
            data_special_day_weights.append(self.init_special_day_weights(sample_data[column_time].tolist()))
            data_neighbor_time_weights.append(
                self.init_neighbor_time_weights(sample_data[column_time].tolist(), second_interval))
            data_labels.append(np.array([sample_label]))
            data_unique_keys.append([sample_unique_key])
            print("\rpace ratio: {}%".format(round((i + 1) / len(self.samples), 6) * 100), end="")
        print("")
        self.data_x = data_x
        self.data_y = data_y
        self.data_x_marks = data_x_marks
        self.data_y_marks = data_y_marks
        self.data_sources = data_sources
        self.data_special_day_weights = data_special_day_weights
        self.data_neighbor_time_weights = data_neighbor_time_weights
        self.data_labels = data_labels
        self.data_unique_keys = data_unique_keys

    def init_special_day_weights(self, time_list):
        """
         get init weight of special day for multichannel calculation
        """
        if self.sample_day_window <= 6:
            # no special day weight for no history day data sample
            return np.zeros(self.seq_len)
        time_list.sort()
        history_times, cur_time = time_list[:-1], time_list[-1]
        cur_datetime = datetime.strptime(cur_time, time_format_input_time)
        cur_date = date(cur_datetime.year, cur_datetime.month, cur_datetime.day)
        calendar = CALENDARS[self.region]
        cur_date_str = "calendar.is_working_day(cur_date)" \
            if self.region != "China" else "calendar.is_workday(cur_date)"
        history_date_str = "calendar.is_working_day(history_date)" \
            if self.region != "China" else "calendar.is_workday(history_date)"
        history_before_date_str = "calendar.is_working_day(history_before_date)" \
            if self.region != "China" else "calendar.is_workday(history_before_date)"
        history_after_date_str = "calendar.is_working_day(history_after_date)" \
            if self.region != "China" else "calendar.is_workday(history_after_date)"
        special_day_weights = []
        for history_time in history_times:
            history_datetime = datetime.strptime(history_time, time_format_input_time)
            history_datetime_before = history_datetime - timedelta(days=1)
            history_datetime_after = history_datetime + timedelta(days=1)
            history_date = date(history_datetime.year, history_datetime.month, history_datetime.day)
            history_before_date = date(
                history_datetime_before.year, history_datetime_before.month, history_datetime_before.day)
            history_after_date = date(
                history_datetime_after.year, history_datetime_after.month, history_datetime_after.day)
            if eval(cur_date_str):
                if eval(history_date_str) and eval(history_before_date_str) and eval(history_after_date_str):
                    special_day_weights.append(complete_weight)
                else:
                    special_day_weights.append(zero_weight)
            else:
                if not eval(history_date_str):
                    special_day_weights.append(complete_weight)
                elif not eval(history_before_date_str) or not eval(history_after_date_str):
                    special_day_weights.append(weak_weight)
                else:
                    special_day_weights.append(zero_weight)
        return np.array(special_day_weights)

    def init_neighbor_time_weights(self, time_list, second_interval):
        """
         get init weight of special day for multichannel calculation
        """
        if self.sample_day_window <= 6:
            # for no history day data sample, neighbor anomaly should be ignored
            return np.array([complete_weight] * (self.sample_time_window_before - self.neighbor_window) + [
                zero_weight] * self.neighbor_window)
        time_list.sort()
        history_times, cur_time = time_list[:-1], time_list[-1]
        cur_date, cur_hour_time = Timer.get_date(cur_time), Timer.get_hour_time(cur_time)
        neighbor_time_weights = []
        for history_time in history_times:
            history_date, history_hour_time = Timer.get_date(history_time), Timer.get_hour_time(history_time)
            if history_date == cur_date:
                # cur day with lower weight
                second_gap = Timer.get_second_gap_between_time(history_time, cur_time)
                if second_gap <= self.neighbor_window * second_interval:
                    neighbor_time_weights.append(zero_weight)
                else:
                    neighbor_time_weights.append(weak_weight)
            else:
                history_tmp_time = cur_date + " " + history_hour_time
                second_gap = Timer.get_second_gap_between_time(history_tmp_time, cur_time)
                if second_gap <= self.neighbor_window * second_interval:
                    neighbor_time_weights.append(complete_weight)
                else:
                    neighbor_time_weights.append(weak_weight)
        return np.array(neighbor_time_weights)

    def get_target_data_and_time_vector(self, source_df):
        """
        get target processed data and time vector
        :return:
        """
        # get target data
        if self.forcast_task == 'M' or self.forcast_task == 'MS':
            cols_data = [col for col in source_df.columns if col != column_time]
            df_data = source_df[cols_data]
        elif self.forcast_task == 'S':
            df_data = source_df[[self.target_index]]
        # scale data
        if self.scale:
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        # get time vector
        df_stamp = source_df[[column_time]]
        df_stamp[column_time] = pd.to_datetime(df_stamp.time)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.time.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.time.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.time.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.time.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop([column_time], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp[column_time].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        return data, data_stamp

    def __getitem__(self, index):
        """
        get iter data
        """
        seq_x = self.data_x[index]
        seq_y = self.data_y[index]
        seq_x_mark = self.data_x_marks[index]
        seq_y_mark = self.data_y_marks[index]
        seq_sources = self.data_sources[index]
        seq_labels = self.data_labels[index]
        seq_special_day_weights = self.data_special_day_weights[index]
        seq_neighbor_time_weights = self.data_neighbor_time_weights[index]
        seq_unique_keys = self.data_unique_keys[index]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_sources, seq_labels, seq_special_day_weights, \
               seq_neighbor_time_weights, seq_unique_keys

    def __len__(self):
        """
        length of dataset
        """
        return len(self.data_x)


if __name__ == '__main__':
    cal = China()
    print(cal.is_working_day(date(2023, 9, 29)))
