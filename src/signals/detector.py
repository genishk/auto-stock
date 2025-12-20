"""신호 탐지 모듈"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from ..discovery.pattern_miner import Pattern, PatternSet


@dataclass
class Signal:
    """매매 신호 데이터 클래스"""
    signal_type: str          # 'BUY' 또는 'SELL'
    ticker: str
    date: str
    pattern_name: str
    confidence: float         # 신뢰도 (0~1)
    expected_return: float    # 예상 수익률 (%)
    holding_period: int       # 권장 보유 기간
    current_price: float
    message: str
    
    def to_dict(self) -> Dict:
        return {
            'signal_type': self.signal_type,
            'ticker': self.ticker,
            'date': self.date,
            'pattern_name': self.pattern_name,
            'confidence': self.confidence,
            'expected_return': self.expected_return,
            'holding_period': self.holding_period,
            'current_price': self.current_price,
            'message': self.message
        }


@dataclass  
class Position:
    """보유 포지션 데이터 클래스"""
    ticker: str
    entry_date: str
    entry_price: float
    pattern_name: str
    holding_period: int       # 목표 보유 기간
    take_profit_pct: float    # 익절 기준 (%)
    stop_loss_pct: float      # 손절 기준 (%)


class SignalDetector:
    """신호 탐지 클래스"""
    
    def __init__(self,
                 confidence_threshold: float = 0.6,
                 take_profit_pct: float = 10,
                 stop_loss_pct: float = -5,
                 max_holding_days: int = 60):
        """
        Args:
            confidence_threshold: 신호 발생 신뢰도 임계값
            take_profit_pct: 익절 기준 (%)
            stop_loss_pct: 손절 기준 (%)
            max_holding_days: 최대 보유 기간
        """
        self.confidence_threshold = confidence_threshold
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_days = max_holding_days
        
        # 현재 보유 포지션 (메모리에 저장, 추후 파일 저장 가능)
        self.positions: Dict[str, Position] = {}
    
    def detect_buy_signal(self,
                         df: pd.DataFrame,
                         features: np.ndarray,
                         pattern_set: PatternSet,
                         top_n_patterns: int = 3) -> Optional[Signal]:
        """
        매수 신호 탐지
        
        Args:
            df: 최신 데이터프레임
            features: 가장 최근 날짜의 특징 벡터 (1D)
            pattern_set: 검증된 패턴 세트
            top_n_patterns: 상위 N개 패턴만 사용
        
        Returns:
            매수 신호 또는 None
        """
        if len(features) == 0:
            return None
        
        # 이미 포지션이 있으면 매수 신호 X
        if pattern_set.ticker in self.positions:
            return None
        
        # 상위 패턴들과 매칭
        best_patterns = pattern_set.get_best_patterns(top_n_patterns)
        
        best_match = None
        best_confidence = 0
        
        for pattern in best_patterns:
            # 유클리드 거리 계산
            distance = np.linalg.norm(features - pattern.centroid)
            
            # 거리를 신뢰도로 변환 (거리가 작을수록 신뢰도 높음)
            # 임계값 2.0 기준, 거리 0이면 신뢰도 1, 거리 2이면 신뢰도 0.5
            confidence = max(0, 1 - distance / 4)
            
            # 패턴의 승률도 반영
            confidence *= pattern.win_rate
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = pattern
        
        # 신뢰도 임계값 체크
        if best_match is None or best_confidence < self.confidence_threshold:
            return None
        
        current_price = df['Close'].iloc[-1]
        current_date = str(df.index[-1].date())
        
        return Signal(
            signal_type='BUY',
            ticker=pattern_set.ticker,
            date=current_date,
            pattern_name=best_match.name,
            confidence=best_confidence,
            expected_return=best_match.avg_return,
            holding_period=pattern_set.holding_period,
            current_price=current_price,
            message=f"🟢 {pattern_set.ticker} 매수 신호! "
                   f"패턴: {best_match.name}, "
                   f"신뢰도: {best_confidence*100:.1f}%, "
                   f"예상수익: {best_match.avg_return:.1f}%"
        )
    
    def detect_sell_signal(self,
                          df: pd.DataFrame,
                          ticker: str) -> Optional[Signal]:
        """
        매도 신호 탐지
        
        Args:
            df: 최신 데이터프레임
            ticker: 종목 티커
        
        Returns:
            매도 신호 또는 None
        """
        if ticker not in self.positions:
            return None
        
        position = self.positions[ticker]
        current_price = df['Close'].iloc[-1]
        current_date = df.index[-1]
        
        # 수익률 계산
        return_pct = (current_price / position.entry_price - 1) * 100
        
        # 보유 기간 계산
        entry_date = pd.to_datetime(position.entry_date)
        days_held = (current_date - entry_date).days
        
        sell_reason = None
        
        # 익절 체크
        if return_pct >= position.take_profit_pct:
            sell_reason = f"익절 ({return_pct:.1f}% 수익)"
        
        # 손절 체크
        elif return_pct <= position.stop_loss_pct:
            sell_reason = f"손절 ({return_pct:.1f}% 손실)"
        
        # 최대 보유 기간 초과
        elif days_held >= self.max_holding_days:
            sell_reason = f"보유기간 초과 ({days_held}일)"
        
        # 목표 보유 기간 도달
        elif days_held >= position.holding_period:
            sell_reason = f"목표 보유기간 도달 ({days_held}일, 수익 {return_pct:.1f}%)"
        
        if sell_reason is None:
            return None
        
        return Signal(
            signal_type='SELL',
            ticker=ticker,
            date=str(current_date.date()),
            pattern_name=position.pattern_name,
            confidence=1.0,  # 매도는 규칙 기반이므로 신뢰도 100%
            expected_return=return_pct,
            holding_period=days_held,
            current_price=current_price,
            message=f"🔴 {ticker} 매도 신호! {sell_reason}"
        )
    
    def add_position(self, signal: Signal) -> None:
        """매수 신호 기반 포지션 추가"""
        if signal.signal_type != 'BUY':
            return
        
        self.positions[signal.ticker] = Position(
            ticker=signal.ticker,
            entry_date=signal.date,
            entry_price=signal.current_price,
            pattern_name=signal.pattern_name,
            holding_period=signal.holding_period,
            take_profit_pct=self.take_profit_pct,
            stop_loss_pct=self.stop_loss_pct
        )
    
    def remove_position(self, ticker: str) -> None:
        """포지션 제거"""
        if ticker in self.positions:
            del self.positions[ticker]
    
    def get_position_status(self, df: pd.DataFrame, ticker: str) -> Optional[Dict]:
        """현재 포지션 상태 조회"""
        if ticker not in self.positions:
            return None
        
        position = self.positions[ticker]
        current_price = df['Close'].iloc[-1]
        current_date = df.index[-1]
        
        return_pct = (current_price / position.entry_price - 1) * 100
        entry_date = pd.to_datetime(position.entry_date)
        days_held = (current_date - entry_date).days
        
        return {
            'ticker': ticker,
            'entry_date': position.entry_date,
            'entry_price': position.entry_price,
            'current_price': current_price,
            'return_pct': return_pct,
            'days_held': days_held,
            'target_days': position.holding_period,
            'pattern': position.pattern_name
        }
    
    def print_status(self, df: pd.DataFrame, ticker: str) -> None:
        """포지션 상태 출력"""
        status = self.get_position_status(df, ticker)
        
        if status is None:
            print(f"\n📭 {ticker}: 보유 포지션 없음")
            return
        
        print(f"\n📊 {ticker} 포지션 현황:")
        print(f"  진입일: {status['entry_date']}")
        print(f"  진입가: ${status['entry_price']:.2f}")
        print(f"  현재가: ${status['current_price']:.2f}")
        print(f"  수익률: {status['return_pct']:+.2f}%")
        print(f"  보유일: {status['days_held']}일 / 목표 {status['target_days']}일")
        print(f"  패턴: {status['pattern']}")

