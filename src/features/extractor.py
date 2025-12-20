"""특징 추출 모듈"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler


class FeatureExtractor:
    """특징 추출 클래스"""
    
    # 패턴 발견에 사용할 특징 컬럼들
    FEATURE_COLUMNS = [
        # 가격 관련
        'returns', 'momentum_10', 'momentum_20',
        'price_vs_ma_short', 'price_vs_ma_medium', 'price_vs_ma_long',
        
        # RSI
        'rsi',
        
        # MACD
        'macd_hist',
        
        # 볼린저 밴드
        'bb_width', 'bb_position',
        
        # 변동성
        'atr_pct', 'volatility_20',
        
        # 거래량
        'volume_ratio',
        
        # 레인지 위치
        'range_position'
    ]
    
    def __init__(self, lookback_period: int = 20):
        """
        Args:
            lookback_period: 패턴 탐지 기간 (수익 케이스 직전 며칠)
        """
        self.lookback_period = lookback_period
        self.scaler = StandardScaler()
        self._is_fitted = False
    
    def extract_features_at_date(self, df: pd.DataFrame, date_idx: int) -> Optional[np.ndarray]:
        """
        특정 날짜 기준 특징 벡터 추출
        
        Args:
            df: 지표가 계산된 데이터프레임
            date_idx: 날짜 인덱스 (정수)
        
        Returns:
            특징 벡터 (1D array) 또는 None
        """
        # lookback 기간만큼의 데이터가 있어야 함
        if date_idx < self.lookback_period:
            return None
        
        # lookback 기간 데이터 추출
        start_idx = date_idx - self.lookback_period
        window_data = df.iloc[start_idx:date_idx]
        
        # 특징 추출
        features = []
        
        for col in self.FEATURE_COLUMNS:
            if col not in df.columns:
                continue
            
            series = window_data[col].dropna()
            if len(series) == 0:
                features.extend([0, 0, 0, 0])  # 기본값
                continue
            
            # 각 특징에 대해 통계값 추출
            features.extend([
                series.iloc[-1],           # 마지막 값 (현재)
                series.mean(),             # 평균
                series.std() if len(series) > 1 else 0,  # 표준편차
                series.iloc[-1] - series.iloc[0] if len(series) > 1 else 0  # 변화량
            ])
        
        return np.array(features)
    
    def extract_features_bulk(self, df: pd.DataFrame, 
                             date_indices: List[int]) -> np.ndarray:
        """
        여러 날짜의 특징 벡터 일괄 추출
        
        Args:
            df: 지표가 계산된 데이터프레임
            date_indices: 날짜 인덱스 리스트
        
        Returns:
            특징 행렬 (2D array: n_samples x n_features)
        """
        features_list = []
        valid_indices = []
        
        for idx in date_indices:
            feat = self.extract_features_at_date(df, idx)
            if feat is not None:
                features_list.append(feat)
                valid_indices.append(idx)
        
        if not features_list:
            return np.array([]), []
        
        return np.array(features_list), valid_indices
    
    def fit_scaler(self, features: np.ndarray) -> None:
        """
        스케일러 학습
        
        Args:
            features: 특징 행렬
        """
        if len(features) > 0:
            # NaN/Inf 처리
            features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)
            self.scaler.fit(features)
            self._is_fitted = True
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        특징 스케일링
        
        Args:
            features: 원본 특징 행렬
        
        Returns:
            스케일링된 특징 행렬
        """
        if not self._is_fitted:
            raise ValueError("Scaler가 학습되지 않았습니다. fit_scaler()를 먼저 호출하세요.")
        
        # NaN/Inf 처리
        features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)
        return self.scaler.transform(features)
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """학습과 변환 동시 수행"""
        self.fit_scaler(features)
        return self.transform(features)
    
    def get_feature_names(self) -> List[str]:
        """특징 이름 목록 반환"""
        names = []
        suffixes = ['_last', '_mean', '_std', '_change']
        
        for col in self.FEATURE_COLUMNS:
            for suffix in suffixes:
                names.append(f"{col}{suffix}")
        
        return names
    
    @property
    def n_features(self) -> int:
        """특징 개수 반환"""
        return len(self.FEATURE_COLUMNS) * 4  # 각 컬럼당 4개 통계값

