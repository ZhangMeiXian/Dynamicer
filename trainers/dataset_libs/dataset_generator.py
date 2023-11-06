# !/usr/bin/python3.8

"""
@ModuleName: module to process and get dataset
@author: zhangmeixian
"""
import numpy as np
import pandas as pd
import warnings
from datetime import date, datetime, timedelta
from workalendar.asia import Singapore, China
from workalendar.europe import Russia
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from trainers.dataset_libs.data_preprocessor import DataPreprocessor
from trainers.methods.net_libs.timefeatures import time_features
from configs.constants import column_time, status_exception, time_format_input_time, complete_weight, weak_weight, \
    zero_weight
from commons.logger import get_logger

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


# config here with your own country
CALENDARS = {
    "China": China(),
    "Russia": Russia(),
    "Singapore": Singapore()
}


class DatasetGenerator(Dataset):
    """
    dataset process and generate
    """

    def __init__(self, forcast_task="S", target_index="data", scale=True, timeenc=0, freq='h', samples=None,
                 sample_time_window_before=30, sample_time_window_after=0, sample_day_window=14, flag=None,
                 region=None, neighbor_window=None):
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
        self.neighbor_window = neighbor_window
        # get the processed dataset
        self.__read_data__()
        logger.info("success to init dataset")

    def __read_data__(self):
        """
        get dataset
        """
        self.scaler = StandardScaler()
        data_x, data_y, data_x_marks, data_y_marks, data_sources, data_labels = [], [], [], [], [], []
        data_special_day_weights, data_neighbor_time_weights = [], []
        # get the data
        for i in range(len(self.samples)):
            sample_obj = self.samples[i]
            sample_label = sample_obj.sample_label
            sample_data = sample_obj.sample_data
            sample_data.columns = [str(x) for x in sample_data.columns]
            if sample_data is None or len(sample_data) <= 0:
                continue
            # data preprocess, filter extraordinary exception data and substitute by median
            sample_data = DataPreprocessor.process_sample_data(
                sample_data, self.sample_time_window_before, self.sample_time_window_after, self.sample_day_window)
            history_data_count = (self.sample_time_window_before + 1 + self.sample_time_window_after) * \
                                 self.sample_day_window + self.sample_time_window_before
            if self.forcast_task == 'M' or self.forcast_task == 'MS':
                cols_data = [col for col in sample_data.columns if col != column_time]
                origin_history_target = sample_data[:history_data_count][cols_data]
            else:
                origin_history_target = sample_data[:history_data_count][[self.target_index]]
            self.scaler.fit(origin_history_target.values)
            target_data, target_marks = self.get_target_data_and_time_vector(sample_data)
            history_data = target_data[:history_data_count]
            history_marks = target_marks[:history_data_count]
            if sample_label == status_exception:
                # get normal predict adjusted value
                end_index = self.sample_time_window_before \
                    if -self.sample_day_window > 0 else -int(self.sample_time_window_before * 1/3)
                start_index = 0 if self.sample_day_window > 0 else int(self.sample_time_window_before * 1/3)
                history_df = pd.DataFrame(history_data[start_index:end_index])
                means = history_df.mean()
                pred_data = pd.DataFrame([means]).values
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
            data_neighbor_time_weights.append(self.init_neighbor_time_weights(sample_data[column_time].tolist()))
            data_labels.append(np.array([sample_label]))
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

    def init_special_day_weights(self, time_list):
        """
         get init weight of special day for multichannel calculation
        """
        time_list.sort()
        if self.sample_day_window <= 0:
            return np.zeros(self.sample_time_window_before)
        history_times, cur_time = time_list[:-1], time_list[-1]
        cur_datetime = datetime.strptime(cur_time, time_format_input_time)
        cur_date = date(cur_datetime.year, cur_datetime.month, cur_datetime.day)
        calendar = CALENDARS[self.region]
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
            if calendar.is_working_day(cur_date):
                if calendar.is_working_day(history_date) and calendar.is_working_day(history_before_date) and \
                        calendar.is_working_day(history_after_date):
                    special_day_weights.append(complete_weight)
                else:
                    special_day_weights.append(zero_weight)
            else:
                if not calendar.is_working_day(history_date):
                    special_day_weights.append(complete_weight)
                elif not calendar.is_working_day(history_before_date) or not calendar.is_working_day(history_after_date):
                    special_day_weights.append(weak_weight)
                else:
                    special_day_weights.append(zero_weight)
        return np.array(special_day_weights)

    def init_neighbor_time_weights(self, time_list):
        """
         get init weight of special day for multichannel calculation
        """
        time_list.sort()


    def get_target_data_and_time_vector(self, source_df):
        """
        根据参数提取处理的数据
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
        获取样本数据
        """
        seq_x = self.data_x[index]
        seq_y = self.data_y[index]
        seq_x_mark = self.data_x_marks[index]
        seq_y_mark = self.data_y_marks[index]
        seq_sources = self.data_sources[index]
        seq_labels = self.data_labels[index]
        seq_special_day_weights = self.data_special_day_weights[index]
        seq_neighbor_time_weights = self.data_neighbor_time_weights[index]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_sources, seq_labels, seq_special_day_weights, \
               seq_neighbor_time_weights

    def __len__(self):
        """
        样本数据的总数
        """
        return len(self.data_x)



if __name__ == '__main__':
    cal = China()
    print(cal.is_working_day(date(2023, 9, 29)))
