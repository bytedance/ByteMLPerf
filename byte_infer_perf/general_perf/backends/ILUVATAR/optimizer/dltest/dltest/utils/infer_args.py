import os

from typing import Union, List, Dict, Any, Mapping
from argparse import Namespace, ArgumentParser
import json


def _obj_to_dict(obj) -> Dict:
    if isinstance(obj, Mapping):
        return obj

    try:
        from absl import flags
        if isinstance(obj, flags.FlagValues):
            return obj.flag_values_dict()
    except:
        pass
    if isinstance(obj, Namespace):
        return obj.__dict__
    elif isinstance(obj, List):
        new_obj = dict()
        for _o in obj:
            _o_dict = _obj_to_dict(_o)
            new_obj.update(_o_dict)
        return new_obj
    elif not isinstance(obj, Dict):
        if hasattr(obj, "__dict__"):
            return obj.__dict__
    try:
        typename = type(obj).__name__
    except:
        typename = str(obj)
    return {typename: str(obj)}


def json_dump_obj(o):
    if hasattr(o, "__name__"):
        return o.__name__
    return str(o)


def show_infer_arguments(args: Union[List, Dict, Any]):
    """ print running arguments
    Example 1: For ArgumentParser
        >>> parser = ArgumentParser("Test")
        >>> parser.add_argument("--arg0", type=str)
        >>> args = parser.parse_args()
        >>> show_infer_arguments(args)

    Example 2: For dict
        >>> args = dict(arg=1)
        >>> show_infer_arguments(args)

    Example 3: For custom object
        >>> from collections import namedtuple
        >>> ArgsType = namedtuple("ArgsType", ["arg"])
        >>> args = ArgsType(arg=123)
        >>> show_infer_arguments(args)

    Example 4: For absl
        >>> from absl import flags
        >>> flags.DEFINE_string("arg", "123", "test")
        >>> show_infer_arguments(flags.FLAGS)

    Example 5: For multi args
        >>> args1 = dict(a=1)
        >>> args2 = dict(b=2)
        >>> show_infer_arguments([args1, args2])

    """
    if not "SHOW_RUNNING_ARGS" in os.environ:
        return

    if os.environ["SHOW_RUNNING_ARGS"].lower() in ["0", "f", "false"]:
        return

    if "LOCAL_RANK" in os.environ:
        if os.environ["LOCAL_RANK"] != "0":
            return
    args = _obj_to_dict(args)
    args = json.dumps(args, default=json_dump_obj)
    print("[RunningArguments]", args)


if __name__ == '__main__':
    os.environ["SHOW_RUNNING_ARGS"] = "1"
    show_infer_arguments([dict(a=1), dict(b=1), object()])