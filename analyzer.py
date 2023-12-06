# -*- coding: UTF-8 -*-
# !/usr/bin/python3.8
"""
@author: zhangmeixian
@module: analyze the results of total models
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from configs.constants import column_time, column_data, column_unqieu_key, column_pred_data, \
    column_recovered_pred_data, column_recovered_data, column_source, column_field

TARGET_PLOT_DETAILS = {
    "SkAB_": {
        "start_index": 0,
        "end_index": 100
    }
}

MODEL_NAME = {
    "adj_ns_Informer": "Informer",
    "adj_ns_Autoformer": "Autoformer",
    "adj_ns_Transformer": "Transformer",
    "adj_ns_FEDformer": "FEDformer",
    "adj_ns_DLinear": "DLinear"
}


class Analyzer(object):
    """
    a class to analyze the results of total models
    """

    @classmethod
    def analyze_total_results(cls):
        """
        analyze the results of total models
        """
        result_root_path = "./results"
        total_datasets = ["SkAB", "NAB", "AIOps", "CSM"]
        for dataset in total_datasets:
            results_df, source_dic = Analyzer.get_total_results_for_each_dataset(result_root_path, dataset)
            results_df[column_time] = results_df[column_unqieu_key].apply(lambda x: x.split("_")[-1])
            results_df[column_field] = results_df[column_unqieu_key].apply(lambda x: x.split("_")[1])
            grouped_results = results_df.groupby(column_field)
            for field, cur_df in grouped_results:
                cur_filed = "_".join([dataset, field])
                start_index = TARGET_PLOT_DETAILS[cur_filed]["start_index"] if cur_filed in TARGET_PLOT_DETAILS else 0
                end_index = TARGET_PLOT_DETAILS[cur_filed]["end_index"] \
                    if cur_filed in TARGET_PLOT_DETAILS else len(cur_df)
                index_count = len(cur_df[column_data].iloc[0])
                cur_df.sort_values(by=column_time, inplace=True)
                not_pred_data_columns = [column_time, column_unqieu_key, column_field, column_data]
                total_columns = cur_df.columns
                for index in range(index_count):
                    cur_df[column_recovered_data + "_index_{}".format(index)] = cur_df.apply(
                        lambda x: Analyzer.get_recovered_data(x.unique_key, x.data, source_dic, index, column_data),
                        axis=1)
                    for column in total_columns:
                        if column not in not_pred_data_columns:
                            cur_df[column + "_recovered_index_{}".format(index)] = cur_df.apply(
                                lambda x: Analyzer.get_recovered_data(x.unique_key, eval("x.{}".format(column)),
                                                                      source_dic, index, column), axis=1)
                plot_columns = [column for column in cur_df.columns if "index_{}".format(index) in column]
                cur_df.sort_values(by=column_time, inplace=True)
                plt.Figure(figsize=(14, 5))
                plot_index = list(range(len(cur_df)))[start_index: end_index]
                plot_times = cur_df[column_time].tolist()[start_index: end_index]
                for plot_column in plot_columns:
                    cur_label = "_".join(plot_column.split("_")[2:4]) if "pred" in plot_column else "True_Data"
                    plot_data = cur_df[plot_column].tolist()[start_index: end_index]
                    plt.plot(plot_index, plot_data, label=cur_label)
                plot_gap = len(plot_times) // 60
                if plot_gap == 0:
                    plot_gap = 10
                plt.xticks(plot_index[::plot_gap], plot_times[::plot_gap], rotation=10)
                plt.legend()
                prefix = "./pics/{}/{}".format(dataset, field.replace("/", "_"))
                if not os.path.exists(prefix):
                    os.makedirs(prefix)
                plt.savefig(prefix + "/index{}_{}_{}.png".format(index, start_index, end_index), dpi=300)
                plt.clf()
                plt.close()
                print()

    @classmethod
    def get_recovered_data(cls, unique_key, data, source_dic, index, column_name=None):
        """
        get the recovered data
        """
        try:
            history_data = source_dic[unique_key]
            # if column_name == column_data:
            #     cur_data = history_data[-1][index]
            #     return cur_data
            history_data = [x[index] for x in history_data]
            history_data = history_data[:-1]
            scaler = StandardScaler()
            scaler.fit(np.array(history_data).reshape(-1, 1))
            cur_data = scaler.transform(np.array([data[index]]).reshape(-1, 1))
            return cur_data.tolist()[0][0]
        except Exception as e:
            return

    @classmethod
    def get_total_results_for_each_dataset(cls, result_root_path, dataset):
        """
        get the total results for each dataset
        """
        total_results, source_dic = pd.DataFrame(), dict()
        for file_root in os.listdir(result_root_path):
            model_name = "_".join(file_root.split("_")[:3])
            model_name = model_name.split("_")[0] \
                if model_name.startswith("tsDynamicer") else MODEL_NAME[model_name]
            file_root = "/".join([result_root_path, file_root])
            for set_name in os.listdir(file_root):
                if set_name == dataset:
                    preds = np.load("/".join([file_root, set_name, "preds.npy"])).squeeze().tolist()
                    trues = np.load("/".join([file_root, set_name, "trues.npy"])).squeeze().tolist()
                    unique_keys = np.load("/".join([file_root, set_name, "unique_keys.npy"])). \
                        reshape(-1, 1).squeeze().tolist()
                    if source_dic is None or len(source_dic) <= 0:
                        cur_source = np.load("/".join([file_root, set_name, "sources.npy"])).squeeze().tolist()
                        for i in range(len(cur_source)):
                            source_dic[unique_keys[i]] = cur_source[i]
                    if len(total_results) <= 0:
                        total_results[column_unqieu_key] = unique_keys
                        total_results[column_data] = trues
                        total_results[column_pred_data + "_{}_{}".format(set_name, model_name)] = preds
                    else:
                        total_results[column_pred_data + "_{}_{}".format(set_name, model_name)] = preds
        return total_results, source_dic


if __name__ == '__main__':
    Analyzer.analyze_total_results()
