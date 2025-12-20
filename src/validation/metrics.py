"""성과 지표 계산 모듈"""

import numpy as np
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """성과 지표 데이터 클래스"""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, 
                               risk_free_rate: float = 0.02,
                               periods_per_year: int = 252) -> float:
        """
        샤프 비율 계산
        
        Args:
            returns: 수익률 배열 (%)
            risk_free_rate: 무위험 수익률 (연간)
            periods_per_year: 연간 기간 수
        
        Returns:
            샤프 비율
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0
        
        # 연간화
        mean_return = returns.mean() * periods_per_year / 100
        std_return = returns.std() * np.sqrt(periods_per_year) / 100
        
        return (mean_return - risk_free_rate) / std_return
    
    @staticmethod
    def calculate_sortino_ratio(returns: np.ndarray,
                                risk_free_rate: float = 0.02,
                                periods_per_year: int = 252) -> float:
        """
        소르티노 비율 계산 (하방 위험만 고려)
        
        Args:
            returns: 수익률 배열 (%)
            risk_free_rate: 무위험 수익률 (연간)
            periods_per_year: 연간 기간 수
        
        Returns:
            소르티노 비율
        """
        if len(returns) == 0:
            return 0
        
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float('inf') if returns.mean() > 0 else 0
        
        mean_return = returns.mean() * periods_per_year / 100
        downside_std = downside_returns.std() * np.sqrt(periods_per_year) / 100
        
        return (mean_return - risk_free_rate) / downside_std
    
    @staticmethod
    def calculate_max_drawdown(returns: np.ndarray) -> float:
        """
        최대 낙폭 계산
        
        Args:
            returns: 수익률 배열 (%)
        
        Returns:
            최대 낙폭 (%)
        """
        if len(returns) == 0:
            return 0
        
        cumulative = (1 + returns / 100).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max * 100
        
        return drawdown.min()
    
    @staticmethod
    def calculate_win_rate(returns: np.ndarray, threshold: float = 0) -> float:
        """
        승률 계산
        
        Args:
            returns: 수익률 배열 (%)
            threshold: 승리 기준 수익률
        
        Returns:
            승률 (0~1)
        """
        if len(returns) == 0:
            return 0
        
        return (returns >= threshold).mean()
    
    @staticmethod
    def calculate_profit_factor(returns: np.ndarray) -> float:
        """
        수익 팩터 계산 (총 이익 / 총 손실)
        
        Args:
            returns: 수익률 배열 (%)
        
        Returns:
            수익 팩터
        """
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        
        if losses == 0:
            return float('inf') if gains > 0 else 0
        
        return gains / losses
    
    @staticmethod
    def calculate_expectancy(returns: np.ndarray) -> float:
        """
        기대값 계산 (거래당 평균 수익)
        
        Args:
            returns: 수익률 배열 (%)
        
        Returns:
            기대값 (%)
        """
        if len(returns) == 0:
            return 0
        
        return returns.mean()
    
    @staticmethod
    def calculate_all(returns: np.ndarray, threshold: float = 0) -> Dict:
        """
        모든 지표 계산
        
        Args:
            returns: 수익률 배열 (%)
            threshold: 승리 기준 수익률
        
        Returns:
            지표 딕셔너리
        """
        returns = np.array(returns)
        
        return {
            'total_trades': len(returns),
            'win_rate': PerformanceMetrics.calculate_win_rate(returns, threshold),
            'avg_return': returns.mean() if len(returns) > 0 else 0,
            'std_return': returns.std() if len(returns) > 0 else 0,
            'total_return': returns.sum(),
            'max_return': returns.max() if len(returns) > 0 else 0,
            'min_return': returns.min() if len(returns) > 0 else 0,
            'sharpe_ratio': PerformanceMetrics.calculate_sharpe_ratio(returns),
            'sortino_ratio': PerformanceMetrics.calculate_sortino_ratio(returns),
            'max_drawdown': PerformanceMetrics.calculate_max_drawdown(returns),
            'profit_factor': PerformanceMetrics.calculate_profit_factor(returns),
            'expectancy': PerformanceMetrics.calculate_expectancy(returns)
        }

