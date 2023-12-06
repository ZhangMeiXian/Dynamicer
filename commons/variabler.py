# !/usr/bin/python3.8

"""
@ModuleName: variable processors
@author: zhangmeixian
"""
import pickle
from commons.logger import get_logger

logger = get_logger(__name__)


class Variabler:
    """
    variable processors
    """

    @classmethod
    def save_variable(cls, data, file_path):
        """
        save large variable to file
        @return:
        """
        f = open(file_path, "wb")
        pickle.dump(data, f)
        f.close()
        logger.info("success to save: {}".format(file_path))
        return file_path

    @classmethod
    def load_variable(cls, file_path):
        """
        load large variable from file
        @return:
        """
        try:
            f = open(file_path, "rb")
            data = pickle.load(f)
            f.close()
            logger.info("success to load: {}".format(file_path))
            return data
        except Exception as e:
            logger.info("failed to load variable for: {}".format(e))
            return None