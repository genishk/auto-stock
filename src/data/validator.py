"""데이터 검증 모듈"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ValidationReport:
    """검증 리포트 데이터 클래스"""
    ticker: str
    total_rows: int
    missing_count: int
    missing_ratio: float
    is_continuous: bool
    price_valid: bool
    issues: list
    
    def is_valid(self) -> bool:
        """데이터가 유효한지 확인"""
        return self.missing_ratio < 0.01 and self.price_valid  # 결측 1% 미만 & 가격 유효


class DataValidator:
    """데이터 검증 클래스"""
    
    @staticmethod
    def validate(df: pd.DataFrame, ticker: str = "UNKNOWN") -> Tuple[pd.DataFrame, ValidationReport]:
        """
        데이터 검증 및 정제
        
        Args:
            df: 원본 데이터프레임
            ticker: 종목 티커
        
        Returns:
            (정제된 데이터프레임, 검증 리포트)
        """
        issues = []
        original_len = len(df)
        
        # 1. 결측치 분석
        missing_count = df[['Open', 'High', 'Low', 'Close']].isnull().any(axis=1).sum()
        missing_ratio = missing_count / len(df) if len(df) > 0 else 0
        
        if missing_count > 0:
            issues.append(f"결측치 {missing_count}개 ({missing_ratio*100:.2f}%)")
        
        # 2. 결측치 처리 (drop)
        df_clean = df.dropna(subset=['Open', 'High', 'Low', 'Close']).copy()
        
        if len(df_clean) < original_len:
            removed = original_len - len(df_clean)
            issues.append(f"결측 행 {removed}개 제거됨")
        
        # 3. 데이터 연속성 검증
        is_continuous = True
        if len(df_clean) > 1:
            date_diff = df_clean.index.to_series().diff()
            # 5일 이상 갭 (공휴일 감안해도 이상)
            gaps = date_diff[date_diff > pd.Timedelta(days=5)]
            if len(gaps) > 0:
                is_continuous = False
                issues.append(f"데이터 갭 {len(gaps)}개 발견")
        
        # 4. 가격 논리 검증
        price_valid = True
        
        # High >= Low
        invalid_hl = df_clean[df_clean['High'] < df_clean['Low']]
        if len(invalid_hl) > 0:
            price_valid = False
            issues.append(f"High < Low: {len(invalid_hl)}건")
        
        # Close 범위 내
        invalid_close = df_clean[
            (df_clean['Close'] > df_clean['High']) | 
            (df_clean['Close'] < df_clean['Low'])
        ]
        if len(invalid_close) > 0:
            price_valid = False
            issues.append(f"Close 범위 벗어남: {len(invalid_close)}건")
        
        # 음수/0 가격
        negative = df_clean[df_clean['Close'] <= 0]
        if len(negative) > 0:
            price_valid = False
            issues.append(f"음수/0 가격: {len(negative)}건")
        
        # 5. 극단적 변동 체크 (50% 이상)
        daily_change = df_clean['Close'].pct_change().abs()
        extreme = df_clean[daily_change > 0.5]
        if len(extreme) > 0:
            issues.append(f"일간 50%+ 변동: {len(extreme)}건 (정상일 수 있음)")
        
        # 리포트 생성
        report = ValidationReport(
            ticker=ticker,
            total_rows=len(df_clean),
            missing_count=missing_count,
            missing_ratio=missing_ratio,
            is_continuous=is_continuous,
            price_valid=price_valid,
            issues=issues
        )
        
        return df_clean, report
    
    @staticmethod
    def validate_all(data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, ValidationReport]]:
        """
        여러 종목 데이터 검증
        
        Args:
            data: {ticker: DataFrame} 딕셔너리
        
        Returns:
            (정제된 데이터 딕셔너리, 리포트 딕셔너리)
        """
        cleaned_data = {}
        reports = {}
        
        for ticker, df in data.items():
            df_clean, report = DataValidator.validate(df, ticker)
            cleaned_data[ticker] = df_clean
            reports[ticker] = report
        
        return cleaned_data, reports

