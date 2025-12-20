"""데이터 캐싱 모듈"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict
import json


class DataCache:
    """데이터 캐싱 클래스"""
    
    def __init__(self, cache_dir: str = "data/cache", max_age_hours: int = 12):
        """
        Args:
            cache_dir: 캐시 디렉토리 경로
            max_age_hours: 캐시 유효 시간 (시간 단위)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age = timedelta(hours=max_age_hours)
        self.metadata_file = self.cache_dir / "metadata.json"
    
    def _get_cache_path(self, ticker: str) -> Path:
        """캐시 파일 경로 반환"""
        return self.cache_dir / f"{ticker}.parquet"
    
    def _load_metadata(self) -> Dict:
        """메타데이터 로드"""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self, metadata: Dict) -> None:
        """메타데이터 저장"""
        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def is_valid(self, ticker: str) -> bool:
        """
        캐시가 유효한지 확인
        
        Args:
            ticker: 종목 티커
        
        Returns:
            캐시가 유효하면 True
        """
        cache_path = self._get_cache_path(ticker)
        
        if not cache_path.exists():
            return False
        
        metadata = self._load_metadata()
        if ticker not in metadata:
            return False
        
        # 캐시 시간 확인
        cached_time = datetime.fromisoformat(metadata[ticker]["cached_at"])
        if datetime.now() - cached_time > self.max_age:
            return False
        
        return True
    
    def get(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        캐시에서 데이터 가져오기
        
        Args:
            ticker: 종목 티커
        
        Returns:
            캐시된 데이터프레임 또는 None
        """
        if not self.is_valid(ticker):
            return None
        
        cache_path = self._get_cache_path(ticker)
        
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"⚠️  {ticker} 캐시 로드 실패: {e}")
            return None
    
    def set(self, ticker: str, df: pd.DataFrame) -> None:
        """
        데이터를 캐시에 저장
        
        Args:
            ticker: 종목 티커
            df: 저장할 데이터프레임
        """
        cache_path = self._get_cache_path(ticker)
        
        try:
            # Parquet으로 저장 (빠르고 용량 작음)
            df.to_parquet(cache_path)
            
            # 메타데이터 업데이트
            metadata = self._load_metadata()
            metadata[ticker] = {
                "cached_at": datetime.now().isoformat(),
                "rows": len(df),
                "start_date": str(df.index[0].date()) if len(df) > 0 else None,
                "end_date": str(df.index[-1].date()) if len(df) > 0 else None
            }
            self._save_metadata(metadata)
            
        except Exception as e:
            print(f"⚠️  {ticker} 캐시 저장 실패: {e}")
    
    def get_last_date(self, ticker: str) -> Optional[datetime]:
        """
        캐시된 데이터의 마지막 날짜 반환
        
        Args:
            ticker: 종목 티커
        
        Returns:
            마지막 날짜 또는 None
        """
        df = self.get(ticker)
        if df is not None and len(df) > 0:
            return df.index[-1].to_pydatetime()
        return None
    
    def update(self, ticker: str, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        기존 캐시에 새 데이터 추가
        
        Args:
            ticker: 종목 티커
            new_data: 새로운 데이터
        
        Returns:
            병합된 데이터프레임
        """
        existing = self.get(ticker)
        
        if existing is None:
            self.set(ticker, new_data)
            return new_data
        
        # 중복 제거하며 병합
        combined = pd.concat([existing, new_data])
        combined = combined[~combined.index.duplicated(keep='last')]
        combined = combined.sort_index()
        
        self.set(ticker, combined)
        return combined
    
    def clear(self, ticker: str = None) -> None:
        """
        캐시 삭제
        
        Args:
            ticker: 종목 티커 (None이면 전체 삭제)
        """
        if ticker:
            cache_path = self._get_cache_path(ticker)
            if cache_path.exists():
                cache_path.unlink()
            
            metadata = self._load_metadata()
            if ticker in metadata:
                del metadata[ticker]
                self._save_metadata(metadata)
        else:
            # 전체 삭제
            for f in self.cache_dir.glob("*.parquet"):
                f.unlink()
            if self.metadata_file.exists():
                self.metadata_file.unlink()
    
    def info(self) -> Dict:
        """캐시 정보 반환"""
        return self._load_metadata()

