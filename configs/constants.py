# !/usr/bin/python3.8

"""
@ModuleName: constants
@author: zhangmeixian
"""

# global column names
column_time = "time"
column_label = "label"
column_data = "data"
column_time_range = "time_range"
column_exception_range = "exception_ranges"
column_anomaly_state = "anomaly_state"
column_source_path = "source_path"
column_unqieu_key = "unique_key"
column_pred_data = "pred_data"
column_recovered_pred_data = "recovered_pred_data"
column_recovered_data = "recovered_data"
column_source = "source"
column_field = "field"


# global digits
status_exception = 1  # status anomalous
status_exception_no = 0  # status normal
label_exception = 1  # labeled anomaly
label_normal = 0  # labeled normal
status_up = "up"
status_down = "down"
status_both = "both"


# time format
time_format_date = "%Y-%m-%d"
time_format_input_time = "%Y-%m-%d %H:%M:%S"
time_format_minute_time = "%Y-%m-%d %H:%M:00"
time_format_hour_time = "%Y-%m-%d %H:00:00"
time_format_model_name = "%Y%m%d%H%M%S"


# digits
complete_weight = 1.0
weak_weight = 0.5
zero_weight = 0.0