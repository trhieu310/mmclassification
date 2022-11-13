# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class Table(CustomDataset):

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.JEPG', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    CLASSES = ["FIGURE", "BBT", "LOGO", "BORDER", "NATURAL", "TABLE"]

    def __init__(self,
                 data_prefix: str,
                 pipeline: Sequence = (),
                 classes: Union[str, Sequence[str], None] = None,
                 ann_file: Optional[str] = None,
                 test_mode: bool = False,
                 file_client_args: Optional[dict] = None):
        super().__init__(
            data_prefix=data_prefix,
            pipeline=pipeline,
            classes=classes,
            ann_file=ann_file,
            extensions=self.IMG_EXTENSIONS,
            test_mode=test_mode,
            file_client_args=file_client_args)
