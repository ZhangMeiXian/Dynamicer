# !/usr/bin/python3.8

"""
@ModuleName: logger
@author: zhangmeixian
"""
import os
import sys
import logging.handlers
from datetime import datetime, timedelta
from configs.constants import time_format_input_time


class Logger(object):
    """
    log to file
    """

    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, "a")

    def write(self, msg):
        """
        write log
        @return:
        """
        self.terminal.write(msg)
        self.log.write(msg)

    def flush(self):
        """
        flush
        """
        pass


def record_log(file_name, file_path="./logs", delete_day_window=7):
    """
    log to file
    @param delete_day_window: history log auto delete window
    @return:
    """
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    time_now_datetime = datetime.now()
    time_now_str = time_now_datetime.strftime(time_format_input_time)
    time_history = (time_now_datetime + timedelta(days=-delete_day_window)).strftime(time_format_input_time)
    saved_file_path = file_path + '/{}_{}.log'.format(file_name, time_now_str)
    sys.stdout = Logger(saved_file_path)
    # delete history log
    logs_list = os.listdir(file_path)
    if len(logs_list) <= 0:
        return
    for file_name in logs_list:
        save_time_str = file_name.split("_")[-1].split(".")[0]
        if save_time_str < time_history:
            # do delete
            os.remove("/".join([file_path, file_name]))

# ANSI escape codes
RED = "\033[1;31m"
YELLOW = "\033[1;33m"
GREEN = "\033[1;32m"
RESET = "\033[0m"


class ColoredFormatter(logging.Formatter):
    """
    color formatter
    """

    COLORS = {
        'WARNING': YELLOW,
        'ERROR': RED,
        'INFO': GREEN,
        'DEBUG': RESET
    }

    def format(self, record):
        """
        format
        """
        color = self.COLORS.get(record.levelname)
        message = super().format(record)
        if color:
            message = color + message + RESET
        return message


def get_logger(name=None):
    """
    get logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = ColoredFormatter("[%(asctime)s] [%(levelname)s] %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger