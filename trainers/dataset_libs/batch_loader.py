# !/usr/bin/python3.8

"""
@ModuleName: batch processor for dataset
@author: zhangmeixian
"""
import os
from torch.utils.data import DataLoader
from trainers.dataset_libs.dataset_generator import DatasetGenerator
from commons.variabler import Variabler
from commons.logger import get_logger

logger = get_logger(__name__)


class BatchLoader(object):
    """
    batch data process
    """

    @classmethod
    def get_dataset(cls, samples, args, flag="train"):
        """
        get batch dataset
        """
        Data = DatasetGenerator
        timeenc = 0 if args.embed != 'timeF' else 1
        # dataset config
        if flag == "test" or flag == "detect":
            shuffle_flag = False
            drop_last = False
            batch_size = 1
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size

        # get dataset
        dataset_path = "./data/dataset/{}/{}_dataset.pickle".format(args.dataset, flag)
        loader_path = "./data/dataset/{}/{}_loader.pickle".format(args.dataset, flag)
        if not os.path.exists(dataset_path):
            data_set = Data(
                forcast_task=args.forcast_task, target_index=args.target_index, scale=True, timeenc=timeenc,
                freq=args.freq, samples=samples, sample_time_window_before=args.sample_time_window_before,
                sample_time_window_after=args.sample_time_window_after, sample_day_window=args.sample_day_window,
                flag=flag, region=args.region, neighbor_window=args.neighbor_window, is_reduce_dim=args.is_reduce_dim,
                target_dim=args.target_dim, seq_len=args.seq_len
            )
            Variabler.save_variable(data_set, dataset_path)
        else:
            data_set = Variabler.load_variable(dataset_path)
        logger.info("{} data count: {}".format(flag, len(data_set)))
        if not os.path.exists(loader_path):
            data_loader = DataLoader(
                data_set, batch_size=batch_size, shuffle=shuffle_flag, num_workers=args.num_workers,
                drop_last=drop_last
            )
            Variabler.save_variable(data_loader, loader_path)
        else:
            data_loader = Variabler.load_variable(loader_path)
        return data_set, data_loader
