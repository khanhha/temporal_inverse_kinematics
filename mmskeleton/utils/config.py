import os
from mmcv import Config as BaseConfig


class Config(BaseConfig):
    @staticmethod
    def fromfile(filename):
        try:
            return BaseConfig.fromfile(filename)
        except:
            from mmskeleton.version import mmskl_home
            return BaseConfig.fromfile(os.path.join(mmskl_home, filename))
