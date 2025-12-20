"""유틸리티 헬퍼 함수들"""

import os
from pathlib import Path
from typing import Any, Dict
import yaml


def get_project_root() -> Path:
    """프로젝트 루트 디렉토리 반환"""
    # 이 파일 위치: src/utils/helpers.py
    # 프로젝트 루트: 2단계 위
    return Path(__file__).parent.parent.parent


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    설정 파일 로드
    
    Args:
        config_path: 설정 파일 경로 (None이면 기본 경로)
    
    Returns:
        설정 딕셔너리
    """
    if config_path is None:
        config_path = get_project_root() / "config" / "settings.yaml"
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config


def ensure_dir(path: Path) -> Path:
    """디렉토리가 없으면 생성"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_pct(value: float, decimals: int = 2) -> str:
    """퍼센트 포맷팅"""
    return f"{value:.{decimals}f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """숫자 포맷팅 (천 단위 쉼표)"""
    return f"{value:,.{decimals}f}"

