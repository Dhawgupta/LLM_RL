from __future__ import annotations
from typing import Dict, Any

class HeadConfig:
    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> HeadConfig:
        return cls(**config_dict)
