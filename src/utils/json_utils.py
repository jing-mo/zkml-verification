import json
import numpy as np
from typing import Any, Dict, List, Union, IO


def convert_to_serializable(obj: Any) -> Any:
    """
    将NumPy类型转换为可JSON序列化的Python原生类型

    Args:
        obj: 需要转换的对象

    Returns:
        转换后的可JSON序列化对象
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    elif hasattr(obj, '__dict__'):
        return convert_to_serializable(obj.__dict__)
    else:
        return obj


class NumpyEncoder(json.JSONEncoder):
    """
    支持NumPy类型的JSON编码器
    """

    def default(self, obj):
        return convert_to_serializable(obj)


def dump_json(obj: Any, fp: IO[str], indent: int = 2, **kwargs) -> None:
    """
    将对象保存为JSON文件，支持NumPy类型

    Args:
        obj: 要保存的对象
        fp: 文件对象
        indent: 缩进空格数
        **kwargs: 传递给json.dump的其他参数
    """
    json.dump(obj, fp, cls=NumpyEncoder, indent=indent, **kwargs)


def load_json(fp: IO[str], **kwargs) -> Any:
    """
    从JSON文件加载对象

    Args:
        fp: 文件对象
        **kwargs: 传递给json.load的其他参数

    Returns:
        加载的对象
    """
    return json.load(fp, **kwargs)


def dumps_json(obj: Any, indent: int = 2, **kwargs) -> str:
    """
    将对象转换为JSON字符串，支持NumPy类型

    Args:
        obj: 要转换的对象
        indent: 缩进空格数
        **kwargs: 传递给json.dumps的其他参数

    Returns:
        JSON字符串
    """
    return json.dumps(obj, cls=NumpyEncoder, indent=indent, **kwargs)


def loads_json(s: str, **kwargs) -> Any:
    """
    从JSON字符串加载对象

    Args:
        s: JSON字符串
        **kwargs: 传递给json.loads的其他参数

    Returns:
        加载的对象
    """
    return json.loads(s, **kwargs)