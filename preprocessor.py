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
    parser.add_argument("--dataset", type=str, nargs="+", required=True, help="Dataset to be preprocessed.")
    args = parser.parse_args()
    DataPreprocessor.execute(args)


if __name__ == '__main__':
    main()



