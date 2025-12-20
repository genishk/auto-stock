"""백테스팅 모듈"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..discovery.pattern_miner import Pattern, PatternSet


@dataclass
class BacktestResult:
    """백테스트 결과 데이터 클래스"""
    pattern_name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_return: float
    total_return: float
    max_return: float
    min_return: float
    std_return: float
    sharpe_ratio: float
    max_drawdown: float
    trade_dates: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'pattern_name': self.pattern_name,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'avg_return': self.avg_return,
            'total_return': self.total_return,
            'max_return': self.max_return,
            'min_return': self.min_return,
            'std_return': self.std_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'trade_dates': self.trade_dates
        }


class Backtester:
    """백테스팅 클래스"""
    
    def __init__(self, 
                 min_win_rate: float = 0.55,
                 min_occurrences: int = 20,
                 min_avg_return: float = 3.0):
        """
        Args:
            min_win_rate: 최소 승률 (검증 통과 기준)
            min_occurrences: 최소 발생 횟수
            min_avg_return: 최소 평균 수익률 (%)
        """
        self.min_win_rate = min_win_rate
        self.min_occurrences = min_occurrences
        self.min_avg_return = min_avg_return
    
    def backtest_pattern(self,
                        pattern: Pattern,
                        df: pd.DataFrame,
                        features: np.ndarray,
                        valid_indices: List[int],
                        pattern_set: PatternSet,
                        threshold: float = 2.0) -> BacktestResult:
        """
        단일 패턴 백테스트
        
        Args:
            pattern: 테스트할 패턴
            df: OHLCV 데이터프레임
            features: 스케일링된 특징 행렬
            valid_indices: 특징에 해당하는 날짜 인덱스
            pattern_set: 패턴 세트 (파라미터 참조용)
            threshold: 패턴 매칭 거리 임계값
        
        Returns:
            BacktestResult 객체
        """
        holding_period = pattern_set.holding_period
        min_return = pattern_set.min_return
        
        # 패턴과 유사한 날짜 찾기
        distances = np.linalg.norm(features - pattern.centroid, axis=1)
        matching_mask = distances < threshold
        matching_indices = np.array(valid_indices)[matching_mask]
        
        # 각 매칭 날짜의 수익률 계산
        returns = []
        trade_dates = []
        
        for idx in matching_indices:
            if idx + holding_period >= len(df):
                continue  # 미래 데이터 부족
            
            entry_price = df['Close'].iloc[idx]
            exit_price = df['Close'].iloc[idx + holding_period]
            ret = (exit_price / entry_price - 1) * 100
            
            returns.append(ret)
            trade_dates.append(str(df.index[idx].date()))
        
        if not returns:
            return BacktestResult(
                pattern_name=pattern.name,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                avg_return=0,
                total_return=0,
                max_return=0,
                min_return=0,
                std_return=0,
                sharpe_ratio=0,
                max_drawdown=0,
                trade_dates=[]
            )
        
        returns = np.array(returns)
        winning = returns >= min_return
        
        # 샤프 비율 계산 (연간화)
        if returns.std() > 0:
            sharpe = (returns.mean() * 12) / (returns.std() * np.sqrt(12))  # 월 단위 가정
        else:
            sharpe = 0
        
        # 최대 낙폭 계산
        cumulative = (1 + returns / 100).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max * 100
        max_dd = drawdown.min()
        
        return BacktestResult(
            pattern_name=pattern.name,
            total_trades=len(returns),
            winning_trades=int(winning.sum()),
            losing_trades=int((~winning).sum()),
            win_rate=winning.mean(),
            avg_return=returns.mean(),
            total_return=returns.sum(),
            max_return=returns.max(),
            min_return=returns.min(),
            std_return=returns.std(),
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            trade_dates=trade_dates
        )
    
    def validate_patterns(self,
                         pattern_set: PatternSet,
                         df: pd.DataFrame,
                         features: np.ndarray,
                         valid_indices: List[int]) -> Tuple[List[Pattern], List[BacktestResult]]:
        """
        패턴 세트 검증
        
        Args:
            pattern_set: 검증할 패턴 세트
            df: 테스트 데이터 (학습에 사용하지 않은 데이터)
            features: 스케일링된 특징 행렬
            valid_indices: 특징에 해당하는 날짜 인덱스
        
        Returns:
            (검증 통과 패턴 리스트, 백테스트 결과 리스트)
        """
        valid_patterns = []
        results = []
        
        for pattern in pattern_set.patterns:
            result = self.backtest_pattern(
                pattern, df, features, valid_indices, pattern_set
            )
            results.append(result)
            
            # 검증 기준 통과 여부
            if (result.win_rate >= self.min_win_rate and
                result.total_trades >= self.min_occurrences and
                result.avg_return >= self.min_avg_return):
                valid_patterns.append(pattern)
        
        return valid_patterns, results
    
    def print_results(self, results: List[BacktestResult]) -> None:
        """백테스트 결과 출력"""
        print("\n" + "="*70)
        print("📊 백테스트 결과")
        print("="*70)
        
        # 승률 기준 정렬
        sorted_results = sorted(results, key=lambda r: r.win_rate, reverse=True)
        
        for result in sorted_results:
            status = "✅" if (result.win_rate >= self.min_win_rate and 
                           result.total_trades >= self.min_occurrences and
                           result.avg_return >= self.min_avg_return) else "❌"
            
            print(f"\n{status} [{result.pattern_name}]")
            print(f"  거래 횟수: {result.total_trades}")
            print(f"  승률: {result.win_rate*100:.1f}%")
            print(f"  평균 수익률: {result.avg_return:.2f}%")
            print(f"  총 수익률: {result.total_return:.2f}%")
            print(f"  샤프 비율: {result.sharpe_ratio:.2f}")
            print(f"  최대 낙폭: {result.max_drawdown:.2f}%")

