import sys
import subprocess
from enum import Enum

__all__ = ["get_iluvatar_card_type", "IluvatarGPU"]

class IluvatarGPU(Enum):
    UNKNOWN = -1
    MR50 = 0
    MR100 = 1
    BI150 = 2

card_ixsmi_names = {
        "BI150": IluvatarGPU.BI150,
        "BI-V150": IluvatarGPU.BI150,
        "MR100": IluvatarGPU.MR100,
        "MR-V100": IluvatarGPU.MR100,
        "MR50": IluvatarGPU.MR50,
        "MR-V50": IluvatarGPU.MR50,
}

def get_iluvatar_card_type():
    command = 'ixsmi -L | grep "GPU \{1,\}0"'
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        for key, value in card_ixsmi_names.items():
            if key in result.stdout:
                return value
        else:
            return IluvatarGPU.UNKNOWN
    else:
        return IluvatarGPU.UNKNOWN
