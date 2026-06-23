from typing import List, Union

from core.chunker.base import BaseChunker
from core.compressor.base import BaseCompressor
from core.filter.base import BaseFilter

PreChunkingModule = Union[BaseFilter, BaseCompressor]


def build_pre_chunking_modules(cfg: dict) -> List[PreChunkingModule]:
    # config의 pre_chunking 리스트를 순서대로 모듈 인스턴스로 변환
    # 슬롯 타입 검증: Filter/Compressor만 허용
    modules = []
    for module_cfg in cfg["pipeline"]["pre_chunking"]:
        module = _build_module(module_cfg)
        if not isinstance(module, (BaseFilter, BaseCompressor)):
            raise TypeError(
                f"pre_chunking 슬롯에는 Filter/Compressor만 들어갈 수 있습니다. "
                f"받은 타입: {type(module).__name__} (config type={module_cfg.get('type')!r})"
            )
        modules.append(module)
    return modules


def build_chunker(cfg: dict) -> BaseChunker:
    # config의 chunker 설정을 Chunker 인스턴스로 변환
    # 슬롯 타입 검증: BaseChunker만 허용
    module = _build_module(cfg["pipeline"]["chunker"])
    if not isinstance(module, BaseChunker):
        raise TypeError(
            f"chunker 슬롯에는 BaseChunker 구현체만 들어갈 수 있습니다. "
            f"받은 타입: {type(module).__name__} (config type={cfg['pipeline']['chunker'].get('type')!r})"
        )
    return module


def build_memory_backend(cfg: dict):
    backend_cfg = cfg.get("memory_backend")
    if backend_cfg is None:
        return None
    return _build_module(backend_cfg)


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

    elif module_type == "AttentionSimilarityChunker":
        from core.chunker.attention_similarity import AttentionSimilarityChunker
        return AttentionSimilarityChunker(**kwargs)

    elif module_type == "NoChunker":
        from core.chunker.no_chunker import NoChunker
        return NoChunker(**kwargs)

    elif module_type == "LLMChunker":
        from core.chunker.llm_chunker import LLMChunker
        return LLMChunker(**kwargs)

    elif module_type == "PrecomputedChunker":
        from core.chunker.precomputed_chunker import PrecomputedChunker
        return PrecomputedChunker(**kwargs)

    elif module_type == "SimpleMemBackend":
        from core.memory.simplemem_backend import SimpleMemBackend
        return SimpleMemBackend(**kwargs)

    elif module_type == "AMemBackend":
        from core.memory.amem_backend import AMemBackend
        return AMemBackend(**kwargs)

    elif module_type == "LLMFilter":
        from core.filter.llm_filter import LLMFilter
        return LLMFilter(**kwargs)

    else:
        raise ValueError(f"Unknown module type: {module_type}")