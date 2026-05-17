from typing import List

from core.chunker.base import BaseChunker
from core.compressor.base import BaseCompressor
from core.filter.base import BaseFilter


def build_pre_chunking_modules(cfg: dict) -> List:
    # config의 pre_chunking 리스트를 순서대로 모듈 인스턴스로 변환
    modules = []
    for module_cfg in cfg["pipeline"]["pre_chunking"]:
        modules.append(_build_module(module_cfg))
    return modules


def build_chunker(cfg: dict) -> BaseChunker:
    # config의 chunker 설정을 Chunker 인스턴스로 변환
    return _build_module(cfg["pipeline"]["chunker"])


def _build_module(module_cfg: dict):
    # type 키를 기준으로 해당 클래스를 import해서 인스턴스 생성
    module_type = module_cfg["type"]
    kwargs = {k: v for k, v in module_cfg.items() if k != "type"}

    if module_type == "NoFilter":
        from core.filter.no_filter import NoFilter
        return NoFilter(**kwargs)

    elif module_type == "NoCompressor":
        from core.compressor.no_compressor import NoCompressor
        return NoCompressor(**kwargs)

    elif module_type == "FixedSizeChunker":
        from core.chunker.fixed_size import FixedSizeChunker
        return FixedSizeChunker(**kwargs)

    else:
        raise ValueError(f"Unknown module type: {module_type}")