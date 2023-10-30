# -*- coding: UTF-8 -*-
# !/usr/bin/python3.8

"""
@ModuleName: 数据集预处理入口
@author: zhangmeixian
"""

import argparse
from trainers.dataset_libs.data_preprocessor import DataPreprocessor


def main():
    """
    执行入口
    """
    parser = argparse.ArgumentParser(description='Preprocessor for data.')
    parser.add_argument("--dataset", type=str, nargs="+", required=True,
                        help="Dataset in [AIOps, MSL, SMAP, NAB, SkAB, SMD, SWaT, WADI, custom] to be preprocessed.")
    # following params must be consistent with train config.
    parser.add_argument("--sample_time_window_before", type=int, default=30,
                        help="Neighbor window before length to process and get samples.")
    parser.add_argument("--sample_time_window_after", type=int, default=0,
                        help="Neighbor window after length to process and get samples.")
    parser.add_argument("--sample_day_window", type=int, default=14, help="History day window for sample to reference.")
    parser.add_argument("--pred_len", type=int, default=1, help="Predict length of sample.")
    args = parser.parse_args()
    DataPreprocessor.execute(args)


if __name__ == '__main__':
    main()