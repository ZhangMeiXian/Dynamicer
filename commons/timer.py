# !/usr/bin/python3.8

"""
@ModuleName: time processor
@author: zhangmeixian
"""
import time
from datetime import datetime, timedelta
from configs.constants import time_format_input_time, time_format_minute_time, time_format_date
from commons.logger import get_logger

logger = get_logger(__name__)


class Timer:
    """
    time processors
    """

    @classmethod
    def format_second_time_to_minute(cls, time_list, minute_interval=1):
        """
        format timestamp with 1s gap to 1min
        """
        rounded_times = []
        current_time = time_list[0]

        # 循环处理时间
        for tmp_time in time_list:
            if (tmp_time - current_time).seconds >= minute_interval * 60:
                # 如果当前时间与前一个时间间隔超过1分钟，更新当前时间
                current_time = tmp_time
            rounded_times.append(current_time)

        # 将结果转换回字符串
        rounded_time_strings = [time.strftime(time_format_minute_time) for time in rounded_times]


    @classmethod
    def get_target_times_by_time_range(cls, minute_interval, time_range, time_format=time_format_minute_time):
        """
        get target time list by time range
        """
        time_list = []
        if time_range is None or len(time_range) <= 0:
            return time_list
        if minute_interval is None:
            return time_list
        for time_range_temp in time_range:
            if len(time_range_temp) != 2:
                continue
            cur_time_list = Timer.get_target_time_list(
                time_range_temp[0], time_range_temp[1], minute_interval, time_format)
            time_list.extend(cur_time_list)
        time_list = list(set(time_list))
        time_list.sort()
        return time_list

    @classmethod
    def get_target_second_time_list(cls, start_time, end_time, second_interval, time_format=time_format_input_time):
        """
        get target time list by second interval
        @return:
        """
        start_time = Timer.str_datetime_format_to_str(start_time, time_format_input_time, time_format)
        end_time = Timer.str_datetime_format_to_str(end_time, time_format_input_time, time_format)
        time_list = [start_time]
        # 包含当前时间（sample_time）
        while start_time < end_time:
            time_list.append(end_time)
            end_time_datetime = datetime.strptime(end_time, time_format) + timedelta(
                seconds=-second_interval)
            end_time = end_time_datetime.strftime(time_format)
            if start_time > end_time:
                break
        return time_list

    @classmethod
    def get_target_time_list(cls, start_time, end_time, minute_interval=60, time_format=time_format_minute_time):
        """
        get target time list by minute interval
        @return:
        """
        start_time = Timer.str_datetime_format_to_str(start_time, time_format_input_time, time_format)
        end_time = Timer.str_datetime_format_to_str(end_time, time_format_input_time, time_format)
        time_list = [start_time]
        # 包含当前时间（sample_time）
        while start_time < end_time:
            time_list.append(end_time)
            end_time_datetime = datetime.strptime(end_time, time_format) + timedelta(
                minutes=-minute_interval)
            end_time = end_time_datetime.strftime(time_format)
            if start_time > end_time:
                break
        return time_list

    @classmethod
    def str_datetime_format_to_str(cls, input_time, origin_format, target_format):
        """
        transform format of time string
        """
        try:
            input_datetime = Timer.str_to_datetime(input_time, origin_format)
            return Timer.datetime_to_str(input_datetime, target_format) if input_datetime is not None else None
        except:
            return None

    @classmethod
    def str_to_timestamp(cls, time_str, time_format=time_format_input_time):
        """
        str to timestamp
        @return:
        """
        time_datetime = time.strptime(time_str, time_format)
        timestamp = time.mktime(time_datetime)
        return int(timestamp)

    @classmethod
    def timestamp_to_str(cls, timestamp_obj, time_format=time_format_input_time):
        """
        timestamp to str
        @return:
        """
        loc_time = time.localtime(timestamp_obj)
        time_str = time.strftime(time_format, loc_time)
        return time_str

    @classmethod
    def get_dynamic_sample_second_time_list(cls, sample_time, second_interval, day_window_length,
                                            before_time_window_length, after_time_window_length, pred_len=1):
        """
        get second time list for dynamic sample
        """
        try:
            sample_time_list = []
            # 获取历史波动范围内的数据时间范围
            for day_gap in range(day_window_length, 0, -1):
                day_time = (datetime.strptime(sample_time, time_format_input_time) - timedelta(
                    days=day_gap)).strftime(time_format_input_time)
                day_time_before = Timer.get_second_time_list_before_time(
                    day_time, time_format_input_time, second_interval, before_time_window_length)
                day_time_after = Timer.get_second_time_list_after_time(
                    day_time, time_format_input_time, second_interval, after_time_window_length)
                sample_time_list = sample_time_list + day_time_before + day_time_after
            # 获取当天波动数据时间范围
            cur_time_list = Timer.get_second_time_list_before_time(
                sample_time, time_format_input_time, second_interval, before_time_window_length)
            # 获取要预测的时间
            if pred_len > 1:
                pred_begin_time = Timer.add_time_by_second(sample_time, second_window_length=second_interval)
                pred_end_time = Timer.add_time_by_second(sample_time,
                                                         second_window_length=second_interval * (pred_len - 1))
                pred_times = Timer.get_target_second_time_list(pred_begin_time, pred_end_time, second_interval)
            else:
                pred_times = []
            sample_time_list = sample_time_list + cur_time_list + pred_times
            return sample_time_list
        except Exception as e:
            logger.info("failed to get sample time list for: {}".format(str(e)))
            return None

    @classmethod
    def get_second_gap_between_time(cls, time1, time2, format=time_format_input_time):
        """
        get second gap between two time
        """
        time1_struct = datetime.strptime(time1, format)
        time2_struct = datetime.strptime(time2, format)
        total_seconds = (time2_struct - time1_struct).total_seconds()
        return abs(total_seconds)

    @classmethod
    def get_dynamic_sample_time_list(cls, sample_time, minute_interval, day_window_length, before_time_window_length,
                                     after_time_window_length, pred_len=1):
        """
        get target sample time by minute interval
        """
        try:
            sample_time_list = []
            # 获取历史波动范围内的数据时间范围
            for day_gap in range(day_window_length, 0, -1):
                day_time = (datetime.strptime(sample_time, time_format_minute_time) - timedelta(
                    days=day_gap)).strftime(time_format_minute_time)
                day_time_before = Timer.get_time_list_before_time(
                    day_time, time_format_minute_time, minute_interval, before_time_window_length)
                day_time_after = Timer.get_time_list_after_time(
                    day_time, time_format_minute_time, minute_interval, after_time_window_length)
                sample_time_list = sample_time_list + day_time_before + day_time_after
            # 获取当天波动数据时间范围
            cur_time_list = Timer.get_time_list_before_time(
                sample_time, time_format_minute_time, minute_interval, before_time_window_length)
            # 获取要预测的时间
            if pred_len > 1:
                pred_begin_time = Timer.add_time_by_minute(sample_time, minute_window_length=minute_interval)
                pred_end_time = Timer.add_time_by_minute(sample_time,
                                                         minute_window_length=minute_interval * (pred_len - 1))
                pred_times = Timer.get_target_time_list(pred_begin_time, pred_end_time, minute_interval)
            else:
                pred_times = []
            sample_time_list = sample_time_list + cur_time_list + pred_times
            return sample_time_list
        except Exception as e:
            logger.info("failed to get sample time list for: {}".format(str(e)))
            return None

    @classmethod
    def get_second_time_list_before_time(cls, sample_time, time_format, second_interval, time_window_count):
        """
        get second time list before sample time
        """
        time_list = []
        if sample_time is None or time_window_count is None or time_window_count <= 0:
            return time_list
        sample_time = datetime.strptime(sample_time, time_format)
        # 包含当前时间（sample_time）
        for i in range(time_window_count, -1, -1):
            cur_time = sample_time - timedelta(seconds=i * second_interval)
            time_list.append(cur_time.strftime(time_format))
        return time_list

    @classmethod
    def get_second_time_list_after_time(cls, sample_time, time_format, second_interval, time_window_count):
        """
        get second time list after sample time
        """
        time_list = []
        if sample_time is None or time_window_count is None or time_window_count <= 0:
            return time_list
        sample_time = datetime.strptime(sample_time, time_format)
        # 不包含当前时间（sample_time）
        for i in range(1, time_window_count + 1):
            cur_time = sample_time + timedelta(seconds=i * second_interval)
            time_list.append(cur_time.strftime(time_format))
        return time_list

    @classmethod
    def get_time_list_before_time(cls, sample_time, time_format, minute_interval, time_window_count):
        """
        get time list before sample time
        """
        time_list = []
        if sample_time is None or time_window_count is None or time_window_count <= 0:
            return time_list
        sample_time = datetime.strptime(sample_time, time_format)
        # 包含当前时间（sample_time）
        for i in range(time_window_count, -1, -1):
            cur_time = sample_time - timedelta(minutes=i * minute_interval)
            time_list.append(cur_time.strftime(time_format))
        return time_list

    @classmethod
    def get_time_list_after_time(cls, sample_time, time_format, minute_interval, time_window_count):
        """
        get time list after sample time
        """
        time_list = []
        if sample_time is None or time_window_count is None or time_window_count <= 0:
            return time_list
        sample_time = datetime.strptime(sample_time, time_format)
        # 不包含当前时间（sample_time）
        for i in range(1, time_window_count + 1):
            cur_time = sample_time + timedelta(minutes=i * minute_interval)
            time_list.append(cur_time.strftime(time_format))
        return time_list

    @classmethod
    def generate_time_list(cls, data_len, minute_interval=1):
        """generate time list"""

        # 设置开始时间
        start_time = datetime.now()
        time_list = []

        for _ in range(data_len):
            # 将时间格式化为字符串，并添加到列表中
            time_list.append(start_time.strftime(time_format_minute_time))
            # 增加m分钟
            start_time += timedelta(minutes=minute_interval)

        return time_list

    @classmethod
    def minus_time_by_minute(cls, time_str, minute_window_length, time_format=time_format_input_time):
        """
        get minused time
        """
        try:
            minused_time = datetime.strptime(time_str, time_format) + timedelta(minutes=-minute_window_length)
            minused_time = minused_time.strftime(time_format)
            return minused_time
        except Exception as e:
            logger.info("failed to minus time by minute for: {}".format(str(e)))
            return None

    @classmethod
    def minus_time_by_day(cls, time_str, day_window_length, time_format=time_format_input_time):
        """
        get minused time by day
        """
        try:
            minused_time = datetime.strptime(time_str, time_format) + timedelta(days=-day_window_length)
            minused_time = minused_time.strftime(time_format)
            return minused_time
        except Exception as e:
            logger.info("failed to minus time by day for: {}".format(str(e)))
            return None

    @classmethod
    def get_current_datetime(cls):
        """
        get current datetime
        """
        return datetime.now()

    @classmethod
    def get_current_datetime_str(cls, time_format=time_format_input_time):
        """
        get current datetime str
        """
        datetime_now = Timer.get_current_datetime()
        return datetime_now.strftime(time_format)

    @classmethod
    def add_time_by_day(cls, time_str, day_window_length, time_format=time_format_input_time):
        """
        get added time by day
        """
        try:
            added_time = datetime.strptime(time_str, time_format) + timedelta(days=day_window_length)
            added_time = added_time.strftime(time_format)
            return added_time
        except Exception as e:
            logger.info("failed to add time by day for: {}".format(str(e)))
            return None

    @classmethod
    def add_time_by_minute(cls, time_str, minute_window_length, time_format=time_format_input_time):
        """
        get added time by minute
        @return:
        """
        try:
            added_time = datetime.strptime(time_str, time_format) + timedelta(minutes=minute_window_length)
            added_time = added_time.strftime(time_format)
            return added_time
        except Exception as e:
            logger.info("failed to add time by minute for: {}".format(str(e)))
            return None

    @classmethod
    def add_time_by_second(cls, time_str, second_window_length, time_format=time_format_input_time):
        """
        get added time by second
        @return:
        """
        try:
            added_time = datetime.strptime(time_str, time_format) + timedelta(seconds=second_window_length)
            added_time = added_time.strftime(time_format)
            return added_time
        except Exception as e:
            logger.info("failed to add time by minute for: {}".format(str(e)))
            return None

    @classmethod
    def str_format_to_target_str(cls, input_time, origin_format, target_format):
        """
        format string
        """
        try:
            input_datetime = Timer.str_to_datetime(input_time, origin_format)
            return Timer.datetime_to_str(input_datetime, target_format) if input_datetime is not None else None
        except:
            return None

    @classmethod
    def str_to_datetime(cls, input_time, origin_format=time_format_input_time):
        """
        str to datetime
        """
        try:
            return datetime.strptime(input_time, origin_format)
        except:
            return None

    @classmethod
    def datetime_to_str(cls, input_time, origin_format):
        """
        datetime to str
        """
        try:
            return input_time.strftime(origin_format)
        except:
            return None

    @classmethod
    def get_minute_interval(cls, second_interval):
        """
        get minute interval
        """
        return int(second_interval / 60) if second_interval is not None else None

    @classmethod
    def format_time_to_minute(cls, input_time):
        """
        format time to minute
        """
        try:
            return datetime.strptime(input_time, time_format_input_time).strftime(time_format_minute_time)
        except Exception:
            return None

    @classmethod
    def str_format_to_target_datetime(cls, input_time, origin_format, target_format):
        """
        foramt string to datetime
        """
        try:
            new_input_time = Timer.str_format_to_target_str(input_time, origin_format, target_format)
            return Timer.str_to_datetime(new_input_time, target_format) if new_input_time is not None else None
        except:
            return None

    @classmethod
    def get_date(cls, time_str):
        """
        get date str
        """
        try:
            return time_str[:10]
        except Exception as e:
            logger.info("failed to get date for: {}".format(str(e)))
            return None

    @classmethod
    def get_hour_time(cls, time_str):
        """
        get hour str
        """
        try:
            return time_str[11:]
        except Exception as e:
            logger.info("failed to get date for: {}".format(str(e)))
            return None
