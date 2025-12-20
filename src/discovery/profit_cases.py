"""수익 케이스 발견 모듈"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass


@dataclass
class ProfitCase:
    """수익 케이스 데이터 클래스"""
    date_idx: int           # 날짜 인덱스
    date: pd.Timestamp      # 날짜
    entry_price: float      # 진입 가격
    exit_price: float       # 청산 가격
    return_pct: float       # 수익률 (%)
    holding_days: int       # 보유 기간


class ProfitCaseFinder:
    """수익 케이스 발견 클래스"""
    
    def __init__(self, 
                 holding_periods: List[int] = [20, 40, 60],
                 min_returns: List[float] = [5, 7, 10]):
        """
        Args:
            holding_periods: 테스트할 보유 기간 리스트 (거래일)
            min_returns: 테스트할 최소 수익률 리스트 (%)
        """
        self.holding_periods = holding_periods
        self.min_returns = min_returns
    
    def find_all_profit_cases(self, df: pd.DataFrame) -> Dict[Tuple[int, float], List[ProfitCase]]:
        """
        모든 (보유기간, 최소수익률) 조합에 대해 수익 케이스 찾기
        
        Args:
            df: OHLCV 데이터프레임
        
        Returns:
            {(holding_period, min_return): [ProfitCase, ...]} 딕셔너리
        """
        results = {}
        
        for holding in self.holding_periods:
            for min_ret in self.min_returns:
                cases = self.find_profit_cases(df, holding, min_ret)
                results[(holding, min_ret)] = cases
        
        return results
    
    def find_profit_cases(self, df: pd.DataFrame, 
                         holding_period: int, 
                         min_return: float) -> List[ProfitCase]:
        """
        특정 조건의 수익 케이스 찾기
        
        Args:
            df: OHLCV 데이터프레임
            holding_period: 보유 기간 (거래일)
            min_return: 최소 수익률 (%)
        
        Returns:
            수익 케이스 리스트
        """
        cases = []
        closes = df['Close'].values
        dates = df.index
        
        # 마지막 holding_period 일은 제외 (미래 데이터 필요)
        for i in range(len(df) - holding_period):
            entry_price = closes[i]
            exit_price = closes[i + holding_period]
            
            return_pct = (exit_price / entry_price - 1) * 100
            
            if return_pct >= min_return:
                cases.append(ProfitCase(
                    date_idx=i,
                    date=dates[i],
                    entry_price=entry_price,
                    exit_price=exit_price,
                    return_pct=return_pct,
                    holding_days=holding_period
                ))
        
        return cases
    
    def analyze_combinations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        모든 조합의 통계 분석
        
        Args:
            df: OHLCV 데이터프레임
        
        Returns:
            조합별 통계 데이터프레임
        """
        all_cases = self.find_all_profit_cases(df)
        
        stats = []
        total_days = len(df)
        
        for (holding, min_ret), cases in all_cases.items():
            n_cases = len(cases)
            
            if n_cases == 0:
                stats.append({
                    'holding_period': holding,
                    'min_return': min_ret,
                    'n_cases': 0,
                    'frequency': 0,
                    'avg_return': 0,
                    'max_return': 0,
                    'std_return': 0
                })
                continue
            
            returns = [c.return_pct for c in cases]
            
            stats.append({
                'holding_period': holding,
                'min_return': min_ret,
                'n_cases': n_cases,
                'frequency': n_cases / (total_days - holding) * 100,  # 발생 빈도 (%)
                'avg_return': np.mean(returns),
                'max_return': np.max(returns),
                'std_return': np.std(returns)
            })
        
        return pd.DataFrame(stats)
    
    def get_best_combination(self, df: pd.DataFrame, 
                            min_cases: int = 50) -> Tuple[int, float, List[ProfitCase]]:
        """
        최적 (보유기간, 최소수익률) 조합 찾기
        
        기준: 케이스 수 >= min_cases 중에서 평균 수익률 최고
        
        Args:
            df: OHLCV 데이터프레임
            min_cases: 최소 케이스 수
        
        Returns:
            (최적_보유기간, 최적_최소수익률, 케이스_리스트)
        """
        stats_df = self.analyze_combinations(df)
        
        # 최소 케이스 수 필터
        valid = stats_df[stats_df['n_cases'] >= min_cases]
        
        if len(valid) == 0:
            # 조건 완화: 가장 많은 케이스
            best_row = stats_df.loc[stats_df['n_cases'].idxmax()]
        else:
            # 평균 수익률 최고
            best_row = valid.loc[valid['avg_return'].idxmax()]
        
        best_holding = int(best_row['holding_period'])
        best_min_ret = float(best_row['min_return'])
        
        # 해당 케이스 반환
        all_cases = self.find_all_profit_cases(df)
        best_cases = all_cases[(best_holding, best_min_ret)]
        
        return best_holding, best_min_ret, best_cases
    
    def summary(self, df: pd.DataFrame) -> None:
        """분석 결과 출력"""
        stats_df = self.analyze_combinations(df)
        
        print("\n" + "="*60)
        print("📊 수익 케이스 분석 결과")
        print("="*60)
        print(f"\n데이터 기간: {len(df)} 거래일")
        print(f"테스트 보유기간: {self.holding_periods}")
        print(f"테스트 최소수익률: {self.min_returns}%")
        
        print("\n[조합별 통계]")
        print(stats_df.to_string(index=False))
        
        best_holding, best_min_ret, best_cases = self.get_best_combination(df)
        print(f"\n✅ 최적 조합: {best_holding}일 보유, {best_min_ret}% 이상")
        print(f"   케이스 수: {len(best_cases)}")
        if best_cases:
            returns = [c.return_pct for c in best_cases]
            print(f"   평균 수익률: {np.mean(returns):.2f}%")

