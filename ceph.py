'''Replace `ImageNet` in torchvision.datasets and `create_dataset` of timm.data'''
import io
from typing import Union, Optional

from mmcv.transforms import LoadImageFromFile as BaseLoadImageFromFile
from mmengine.fileio import get_file_backend
from mmpretrain.datasets import ImageNet, ImageNet21k
from PIL import Image

__all__ = ['MMImageNet', 'MMImageNet21k', 'create_dataset']

class MMImageNet(ImageNet):
    """用以替换 `torchvision.datasets`  的 `ImageNet` 和 `ImageFolder`.

    Note:
        在具体使用时, 需要传入 transform 参数

    Example:

        >>> from dataset import MMImageNet
        >>> # 读本地的文件夹形式,
        >>> train = MMImageNet(data_root="/mnt/petrelfs/openmmlab/datasets/classificaton/imagenet/", split='train')
        >>> len(train)
        1281167
        >>> train[12131]
        (<PIL.Image.Image image mode=RGB size=500x333 at 0x7F46D9062250>, 9)
        >>>
        >>> # 读 ceph 上的标注格式 (s 集群上默认格式)
        >>> val = MMImageNet(data_root="/mnt/petrelfs/openmmlab/datasets/classificaton/imagenet/", split='val')
        >>> len(val)
        50000
        >>> val[1000]
        (<PIL.Image.Image image mode=RGB size=500x317 at 0x7F47FC5B1310>, 188)
    """

    def __init__(self,
                 data_root,
                 split,
                 transform = None,
                 data_prefix: Union[str, dict] = '',
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 **kwargs):
        super().__init__(
            data_root=data_root,
            split=split,
            ann_file=ann_file,
            data_prefix=data_prefix,
            metainfo=metainfo,
            **kwargs)
        self.transform = transform
        backend = get_file_backend(data_root)
        self.load = PILLoadImageFromFile(backend=backend)

    def __getitem__(self, idx: int) -> dict:
        result = super().__getitem__(idx)
        result = self.load(result)
        img, label = result['img'], result['gt_label']
        img = img.convert("RGB")
        try:
            if self.transform is not None:
                img = self.transform(img)
        except RuntimeError as e:
            print(result)
            raise e
        return img, int(label)


class MMImageNet21k(ImageNet21k):
    """用以替换 `torchvision.datasets`  的 `ImageNet` 和 `ImageFolder`.

    Note:
        在具体使用时, 需要传入 transform 参数

    Example:

        >>> from dataset import MMImageNet21k
        >>> # 读ceph上的数据集
        >>> train = MMImageNet21k(data_prefix="openmmlab:s3://openmmlab/datasets/classification/imagenet22k/train")
        >>> train
        Dataset MMImageNet21k
            Number of samples:  6605621
            Number of categories:       10122
            Annotation file: 
            Prefix of images:   openmmlab:s3://openmmlab/datasets/classification/imagenet22k/train
        >>> train[12131]
        (<PIL.Image.Image image mode=RGB size=350x254>, 0)
        >>> # 读ceph上的数据集
        >>> train = MMImageNet21k(data_prefix="opendatalab:s3://odl-flat/ImageNet-21k/")
        >>> len(train)
        >>> train[12131]
        (<PIL.Image.Image image mode=RGB size=500x333 at 0x7FDDAC778550>, 9)
    """

    def __init__(self,
                 data_root='',
                 split='',
                 transform = None,
                 data_prefix: Union[str, dict] = '',
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 **kwargs):
        super().__init__(
            data_root=data_root,
            split=split,
            ann_file=ann_file,
            data_prefix=data_prefix,
            metainfo=metainfo,
            **kwargs)
        self.transform = transform
        backend = get_file_backend(data_root)
        self.load = PILLoadImageFromFile(backend=backend)

    def __getitem__(self, idx: int) -> dict:
        result = super().__getitem__(idx)
        result = self.load(result)
        img, label = result['img'], result['gt_label']
        img = img.convert("RGB")
        try:
            if self.transform is not None:
                img = self.transform(img)
        except RuntimeError as e:
            print(result)
            raise e
        return img, int(label)

def create_dataset(
        name,
        root,
        ann_file="",
        split='val',
        search_split=True,
        class_map=None,
        load_bytes=False,
        is_training=False,
        download=False,
        batch_size=None,
        repeats=0,
        **kwargs
):
    """ 使timm.data.create_dataset支持 `MMClsImageNet`. 其本身保持timm的用法

    Note:
        在具体使用时, 不需要传入 transform 参数, timm 在 ``create_loader`` 时 实例化 transform

    Example:

        >>> from dataset import create_dataset
        >>> # 读本地的文件夹形式
        >>> dataset_train = create_dataset("MMImageNet", '/mnt/petrelfs/openmmlab/datasets/classificaton/imagenet/', split='train')
        >>> len(dataset_train)
        1281167
        >>> dataset_train[12131]
        (<PIL.Image.Image image mode=RGB size=500x333 at 0x7F46F50CE3A0>, 9)
        >>>
        >>> # 读 ceph 上的标注格式 (s 集群上默认格式)
        >>> dataset_val = create_dataset("MMImageNet", '/mnt/petrelfs/openmmlab/datasets/classificaton/imagenet/', split='val')
        >>> len(dataset_val)
        50000
        >>> dataset_val[1000]
        (<PIL.Image.Image image mode=RGB size=500x317 at 0x7F46F4D0FDF0>, 188)
    """
    name = name.lower()
    if name == 'mmimagenet':
        ds = MMImageNet(data_root=root, split=split, **kwargs)
    else:
        try:
            import timm
        except:
            raise ImportError("Please install timm")
        ds = timm.data.create_dataset(name,
                                    root,
                                    split=split,
                                    search_split=search_split,
                                    class_map=class_map,
                                    load_bytes=load_bytes,
                                    is_training=is_training,
                                    download=download,
                                    batch_size=batch_size,
                                    repeats=repeats,
                                    **kwargs)
    return ds


class PILLoadImageFromFile(BaseLoadImageFromFile):

    def __init__(self,
                 backend,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.backend = backend
        

    def transform(self, results: dict):
        filename = results['img_path']
        try:
            img_bytes = self.backend.get(filename)
            buff = io.BytesIO(img_bytes)
            results['img'] = Image.open(buff)

        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e

        results['img_shape'] = results['img'].size
        results['ori_shape'] = results['img'].size
        return results
