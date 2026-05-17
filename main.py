import argparse

import yaml

from data.loader import load_locomo10
from data.schema import ProcessedSample
from factory import build_chunker, build_pre_chunking_modules


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(cfg: dict):
    # 1. 데이터 로드 (현재 entrypoint는 단일 샘플 전용. 전체 실행은 추후 별도 entrypoint로 분리)
    dataset_cfg = cfg["dataset"]
    sample_idx = dataset_cfg.get("sample_idx", 0)
    if not isinstance(sample_idx, int):
        raise ValueError(
            f"sample_idx must be an integer (full-dataset run not supported here), got {sample_idx!r}"
        )
    raw = load_locomo10(dataset_cfg["path"], sample_idx=sample_idx)
    print(f"[loader] sample_id: {raw.sample_id}")
    print(f"[loader] num turns: {len(raw.turns)}")
    print(f"[loader] num qa: {len(raw.qa)}")

    # 2. 파이프라인 실행
    pre_chunking_modules = build_pre_chunking_modules(cfg)
    chunker = build_chunker(cfg)

    turns = raw.turns
    for module in pre_chunking_modules:
        turns = module.run(turns)
    print(f"[pipeline] turns after pre-chunking: {len(turns)}")

    chunks = chunker.chunk(turns)
    print(f"[chunker] num chunks: {len(chunks)}")
    if chunks:
        print(f"[chunker] first chunk text:\n{chunks[0].text}")
    else:
        print("[chunker] (no chunks produced — pre-chunking이 모든 turn을 제거함)")
    print()

    # 3. ProcessedSample 생성
    module_names = [m["type"] for m in cfg["pipeline"]["pre_chunking"]]
    module_names.append(cfg["pipeline"]["chunker"]["type"])

    processed = ProcessedSample(
        sample_id=raw.sample_id,
        chunks=chunks,
        qa=raw.qa,
        metadata={"pipeline": module_names},
    )
    print(f"[processed] sample_id: {processed.sample_id}")
    print(f"[processed] num chunks: {len(processed.chunks)}")
    print(f"[processed] first qa: {processed.qa[0]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="config yaml 파일 경로")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()