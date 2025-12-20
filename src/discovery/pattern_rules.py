"""
규칙 기반 패턴 정의 모듈
- 클러스터링 대신 명확한 규칙으로 패턴 정의
- 해석 가능하고 과적합 위험 낮음
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class PatternMatch:
    """패턴 매칭 결과"""
    pattern_name: str
    date_idx: int
    date: pd.Timestamp
    confidence: float  # 0~1
    details: Dict[str, float]  # 매칭 상세 (각 조건 값)


class BasePattern(ABC):
    """패턴 베이스 클래스"""
    
    name: str = "BasePattern"
    description: str = ""
    
    @abstractmethod
    def check(self, df: pd.DataFrame, idx: int) -> Optional[PatternMatch]:
        """
        특정 날짜에 패턴이 발생했는지 확인
        
        Args:
            df: 지표가 계산된 데이터프레임
            idx: 확인할 날짜 인덱스
        
        Returns:
            PatternMatch 또는 None
        """
        pass
    
    def scan(self, df: pd.DataFrame, start_idx: int = 0) -> List[PatternMatch]:
        """전체 데이터에서 패턴 스캔"""
        matches = []
        for idx in range(start_idx, len(df)):
            match = self.check(df, idx)
            if match:
                matches.append(match)
        return matches


# =============================================================================
# 규칙 기반 패턴들
# =============================================================================

class RSIOversold(BasePattern):
    """RSI 과매도 반등 패턴"""
    
    name = "RSI_Oversold"
    description = "RSI가 과매도 구간에서 반등할 때"
    
    def __init__(self, oversold_threshold: float = 30, recovery_threshold: float = 35):
        self.oversold_threshold = oversold_threshold
        self.recovery_threshold = recovery_threshold
    
    def check(self, df: pd.DataFrame, idx: int) -> Optional[PatternMatch]:
        if idx < 5 or 'rsi' not in df.columns:
            return None
        
        rsi_current = df['rsi'].iloc[idx]
        rsi_prev = df['rsi'].iloc[idx-1]
        rsi_min_5d = df['rsi'].iloc[idx-5:idx].min()
        
        # 조건: 최근 5일 내 과매도 + 현재 반등 중
        if (rsi_min_5d < self.oversold_threshold and 
            rsi_prev < self.recovery_threshold and 
            rsi_current > rsi_prev):
            
            confidence = min(1.0, (self.oversold_threshold - rsi_min_5d) / 10)
            
            return PatternMatch(
                pattern_name=self.name,
                date_idx=idx,
                date=df.index[idx],
                confidence=confidence,
                details={
                    'rsi_current': rsi_current,
                    'rsi_min_5d': rsi_min_5d
                }
            )
        return None


class BollingerSqueeze(BasePattern):
    """볼린저 밴드 스퀴즈 후 상단 돌파 패턴"""
    
    name = "BB_Squeeze_Breakout"
    description = "볼린저밴드 수축 후 상단 돌파"
    
    def __init__(self, squeeze_percentile: float = 20, lookback: int = 20):
        self.squeeze_percentile = squeeze_percentile
        self.lookback = lookback
    
    def check(self, df: pd.DataFrame, idx: int) -> Optional[PatternMatch]:
        if idx < self.lookback or 'bb_width' not in df.columns:
            return None
        
        bb_width_current = df['bb_width'].iloc[idx]
        bb_width_history = df['bb_width'].iloc[idx-self.lookback:idx]
        squeeze_threshold = np.percentile(bb_width_history, self.squeeze_percentile)
        
        bb_position = df['bb_position'].iloc[idx]
        bb_position_prev = df['bb_position'].iloc[idx-1]
        
        # 조건: 밴드폭 하위 20% + 상단 돌파 (position > 1)
        if (bb_width_current < squeeze_threshold and 
            bb_position > 0.8 and 
            bb_position > bb_position_prev):
            
            confidence = min(1.0, bb_position - 0.5)
            
            return PatternMatch(
                pattern_name=self.name,
                date_idx=idx,
                date=df.index[idx],
                confidence=confidence,
                details={
                    'bb_width': bb_width_current,
                    'bb_position': bb_position,
                    'squeeze_threshold': squeeze_threshold
                }
            )
        return None


class GoldenCross(BasePattern):
    """골든크로스 패턴 (단기 MA > 장기 MA 돌파)"""
    
    name = "Golden_Cross"
    description = "단기 이동평균이 장기 이동평균을 상향 돌파"
    
    def __init__(self, short_period: int = 20, long_period: int = 50):
        self.short_period = short_period
        self.long_period = long_period
    
    def check(self, df: pd.DataFrame, idx: int) -> Optional[PatternMatch]:
        if idx < 2 or 'ma_short' not in df.columns or 'ma_medium' not in df.columns:
            return None
        
        ma_short = df['ma_short'].iloc[idx]
        ma_long = df['ma_medium'].iloc[idx]
        ma_short_prev = df['ma_short'].iloc[idx-1]
        ma_long_prev = df['ma_medium'].iloc[idx-1]
        
        # 조건: 어제는 단기 < 장기, 오늘은 단기 > 장기
        if (ma_short_prev <= ma_long_prev and ma_short > ma_long):
            
            confidence = min(1.0, (ma_short - ma_long) / ma_long * 100)
            
            return PatternMatch(
                pattern_name=self.name,
                date_idx=idx,
                date=df.index[idx],
                confidence=confidence,
                details={
                    'ma_short': ma_short,
                    'ma_long': ma_long,
                    'cross_strength': (ma_short - ma_long) / ma_long * 100
                }
            )
        return None


class VolumeSpike(BasePattern):
    """거래량 급증 + 가격 상승 패턴"""
    
    name = "Volume_Spike_Up"
    description = "거래량 급증과 함께 가격 상승"
    
    def __init__(self, volume_multiplier: float = 2.0, min_price_change: float = 0.5):
        self.volume_multiplier = volume_multiplier
        self.min_price_change = min_price_change
    
    def check(self, df: pd.DataFrame, idx: int) -> Optional[PatternMatch]:
        if idx < 1 or 'volume_ratio' not in df.columns:
            return None
        
        volume_ratio = df['volume_ratio'].iloc[idx]
        price_change = df['returns'].iloc[idx] * 100 if 'returns' in df.columns else 0
        
        # 조건: 거래량 2배 이상 + 가격 상승
        if (volume_ratio >= self.volume_multiplier and 
            price_change >= self.min_price_change):
            
            confidence = min(1.0, volume_ratio / 3)
            
            return PatternMatch(
                pattern_name=self.name,
                date_idx=idx,
                date=df.index[idx],
                confidence=confidence,
                details={
                    'volume_ratio': volume_ratio,
                    'price_change': price_change
                }
            )
        return None


class MACDCrossover(BasePattern):
    """MACD 골든크로스 패턴"""
    
    name = "MACD_Crossover"
    description = "MACD가 시그널선을 상향 돌파"
    
    def check(self, df: pd.DataFrame, idx: int) -> Optional[PatternMatch]:
        if idx < 2 or 'macd' not in df.columns or 'macd_signal' not in df.columns:
            return None
        
        macd = df['macd'].iloc[idx]
        signal = df['macd_signal'].iloc[idx]
        macd_prev = df['macd'].iloc[idx-1]
        signal_prev = df['macd_signal'].iloc[idx-1]
        
        # 조건: 어제 MACD < Signal, 오늘 MACD > Signal
        if (macd_prev <= signal_prev and macd > signal):
            
            hist = df['macd_hist'].iloc[idx]
            confidence = min(1.0, abs(hist) * 10)
            
            return PatternMatch(
                pattern_name=self.name,
                date_idx=idx,
                date=df.index[idx],
                confidence=confidence,
                details={
                    'macd': macd,
                    'signal': signal,
                    'histogram': hist
                }
            )
        return None


class PriceAtSupport(BasePattern):
    """20일 저점 근처 지지 패턴"""
    
    name = "Price_At_Support"
    description = "가격이 20일 저점 근처에서 반등"
    
    def __init__(self, proximity_pct: float = 2.0):
        self.proximity_pct = proximity_pct
    
    def check(self, df: pd.DataFrame, idx: int) -> Optional[PatternMatch]:
        if idx < 20:
            return None
        
        close = df['Close'].iloc[idx]
        low_20 = df['Low'].iloc[idx-20:idx].min()
        proximity = (close - low_20) / low_20 * 100
        
        # 오늘 상승 중인지
        returns = (close / df['Close'].iloc[idx-1] - 1) * 100
        
        # 조건: 20일 저점 근처 + 반등 중
        if proximity <= self.proximity_pct and returns > 0:
            
            confidence = min(1.0, (self.proximity_pct - proximity) / self.proximity_pct)
            
            return PatternMatch(
                pattern_name=self.name,
                date_idx=idx,
                date=df.index[idx],
                confidence=confidence,
                details={
                    'close': close,
                    'low_20': low_20,
                    'proximity_pct': proximity,
                    'daily_return': returns
                }
            )
        return None


class MomentumReversal(BasePattern):
    """모멘텀 반전 패턴 (하락 후 상승 전환)"""
    
    name = "Momentum_Reversal"
    description = "하락 모멘텀에서 상승 모멘텀으로 전환"
    
    def check(self, df: pd.DataFrame, idx: int) -> Optional[PatternMatch]:
        if idx < 5 or 'momentum_10' not in df.columns:
            return None
        
        mom_current = df['momentum_10'].iloc[idx]
        mom_prev = df['momentum_10'].iloc[idx-1]
        mom_5d_ago = df['momentum_10'].iloc[idx-5]
        
        # 조건: 5일 전 음수 모멘텀 → 현재 양수 전환 중
        if (mom_5d_ago < -3 and  # 5일 전 -3% 이하
            mom_prev < 0 and     # 어제까지 음수
            mom_current > mom_prev):  # 상승 전환
            
            confidence = min(1.0, abs(mom_5d_ago) / 10)
            
            return PatternMatch(
                pattern_name=self.name,
                date_idx=idx,
                date=df.index[idx],
                confidence=confidence,
                details={
                    'momentum_current': mom_current,
                    'momentum_5d_ago': mom_5d_ago
                }
            )
        return None


# =============================================================================
# 추가 패턴들 (다양한 관점)
# =============================================================================

class StochasticOversold(BasePattern):
    """스토캐스틱 과매도 반등 패턴"""
    
    name = "Stochastic_Oversold"
    description = "스토캐스틱 K가 과매도 구간에서 반등"
    
    def __init__(self, oversold_threshold: float = 20, k_period: int = 14):
        self.oversold_threshold = oversold_threshold
        self.k_period = k_period
    
    def check(self, df: pd.DataFrame, idx: int) -> Optional[PatternMatch]:
        if idx < self.k_period + 5:
            return None
        
        # 스토캐스틱 K 계산
        low_min = df['Low'].iloc[idx-self.k_period+1:idx+1].min()
        high_max = df['High'].iloc[idx-self.k_period+1:idx+1].max()
        
        if high_max == low_min:
            return None
        
        stoch_k = (df['Close'].iloc[idx] - low_min) / (high_max - low_min) * 100
        
        # 이전 값
        low_min_prev = df['Low'].iloc[idx-self.k_period:idx].min()
        high_max_prev = df['High'].iloc[idx-self.k_period:idx].max()
        stoch_k_prev = (df['Close'].iloc[idx-1] - low_min_prev) / (high_max_prev - low_min_prev) * 100 if high_max_prev != low_min_prev else 50
        
        # 조건: 과매도에서 반등
        if stoch_k_prev < self.oversold_threshold and stoch_k > stoch_k_prev:
            confidence = min(1.0, (self.oversold_threshold - stoch_k_prev) / 20)
            
            return PatternMatch(
                pattern_name=self.name,
                date_idx=idx,
                date=df.index[idx],
                confidence=confidence,
                details={
                    'stoch_k': stoch_k,
                    'stoch_k_prev': stoch_k_prev
                }
            )
        return None


class WilliamsROversold(BasePattern):
    """Williams %R 과매도 반등 패턴"""
    
    name = "Williams_R_Oversold"
    description = "Williams %R이 과매도 구간(-80 이하)에서 반등"
    
    def __init__(self, oversold_threshold: float = -80, period: int = 14):
        self.oversold_threshold = oversold_threshold
        self.period = period
    
    def check(self, df: pd.DataFrame, idx: int) -> Optional[PatternMatch]:
        if idx < self.period + 2:
            return None
        
        # Williams %R 계산
        high_max = df['High'].iloc[idx-self.period+1:idx+1].max()
        low_min = df['Low'].iloc[idx-self.period+1:idx+1].min()
        
        if high_max == low_min:
            return None
        
        williams_r = (high_max - df['Close'].iloc[idx]) / (high_max - low_min) * -100
        
        # 이전 값
        high_max_prev = df['High'].iloc[idx-self.period:idx].max()
        low_min_prev = df['Low'].iloc[idx-self.period:idx].min()
        williams_r_prev = (high_max_prev - df['Close'].iloc[idx-1]) / (high_max_prev - low_min_prev) * -100 if high_max_prev != low_min_prev else -50
        
        # 조건: 과매도에서 반등
        if williams_r_prev < self.oversold_threshold and williams_r > williams_r_prev:
            confidence = min(1.0, (self.oversold_threshold - williams_r_prev) / 20)
            
            return PatternMatch(
                pattern_name=self.name,
                date_idx=idx,
                date=df.index[idx],
                confidence=confidence,
                details={
                    'williams_r': williams_r,
                    'williams_r_prev': williams_r_prev
                }
            )
        return None


class ATRBreakout(BasePattern):
    """ATR 변동성 돌파 패턴"""
    
    name = "ATR_Breakout"
    description = "전일 종가 + ATR 돌파 (변동성 돌파)"
    
    def __init__(self, atr_period: int = 14, multiplier: float = 0.5):
        self.atr_period = atr_period
        self.multiplier = multiplier
    
    def check(self, df: pd.DataFrame, idx: int) -> Optional[PatternMatch]:
        if idx < self.atr_period + 2:
            return None
        
        # ATR 계산
        tr_list = []
        for i in range(idx - self.atr_period + 1, idx + 1):
            high_low = df['High'].iloc[i] - df['Low'].iloc[i]
            high_close = abs(df['High'].iloc[i] - df['Close'].iloc[i-1])
            low_close = abs(df['Low'].iloc[i] - df['Close'].iloc[i-1])
            tr_list.append(max(high_low, high_close, low_close))
        
        atr = np.mean(tr_list)
        
        prev_close = df['Close'].iloc[idx-1]
        current_high = df['High'].iloc[idx]
        breakout_level = prev_close + atr * self.multiplier
        
        # 조건: 오늘 고가가 돌파 레벨 돌파 + 종가 유지
        if current_high > breakout_level and df['Close'].iloc[idx] > prev_close:
            confidence = min(1.0, (df['Close'].iloc[idx] - prev_close) / atr)
            
            return PatternMatch(
                pattern_name=self.name,
                date_idx=idx,
                date=df.index[idx],
                confidence=confidence,
                details={
                    'atr': atr,
                    'breakout_level': breakout_level,
                    'current_close': df['Close'].iloc[idx]
                }
            )
        return None


class BullishEngulfing(BasePattern):
    """상승 장악형 캔들 패턴"""
    
    name = "Bullish_Engulfing"
    description = "하락 캔들을 상승 캔들이 완전히 감싸는 패턴"
    
    def check(self, df: pd.DataFrame, idx: int) -> Optional[PatternMatch]:
        if idx < 2:
            return None
        
        # 어제: 하락 캔들
        prev_open = df['Open'].iloc[idx-1]
        prev_close = df['Close'].iloc[idx-1]
        prev_body = prev_close - prev_open
        
        # 오늘: 상승 캔들
        curr_open = df['Open'].iloc[idx]
        curr_close = df['Close'].iloc[idx]
        curr_body = curr_close - curr_open
        
        # 조건: 어제 하락 + 오늘 상승 + 오늘이 어제를 감쌈
        if (prev_body < 0 and  # 어제 하락
            curr_body > 0 and  # 오늘 상승
            curr_open <= prev_close and  # 오늘 시가 <= 어제 종가
            curr_close >= prev_open):     # 오늘 종가 >= 어제 시가
            
            # 신뢰도: 오늘 몸통이 어제보다 클수록 높음
            size_ratio = abs(curr_body) / abs(prev_body) if prev_body != 0 else 1
            confidence = min(1.0, size_ratio / 2)
            
            return PatternMatch(
                pattern_name=self.name,
                date_idx=idx,
                date=df.index[idx],
                confidence=confidence,
                details={
                    'prev_body': prev_body,
                    'curr_body': curr_body,
                    'size_ratio': size_ratio
                }
            )
        return None


class DoubleBottom(BasePattern):
    """이중 바닥 패턴 (W 패턴)"""
    
    name = "Double_Bottom"
    description = "비슷한 저점이 2번 형성 후 반등"
    
    def __init__(self, lookback: int = 30, tolerance_pct: float = 2.0):
        self.lookback = lookback
        self.tolerance_pct = tolerance_pct
    
    def check(self, df: pd.DataFrame, idx: int) -> Optional[PatternMatch]:
        if idx < self.lookback + 5:
            return None
        
        # 최근 lookback 기간의 저가들
        lows = df['Low'].iloc[idx-self.lookback:idx+1]
        
        # 두 개의 저점 찾기
        min1_idx = lows.idxmin()
        min1_val = lows[min1_idx]
        
        # 첫 번째 저점 주변 제외하고 두 번째 저점 찾기
        min1_pos = lows.index.get_loc(min1_idx)
        
        # 최소 5일 떨어진 두 번째 저점
        mask = np.abs(np.arange(len(lows)) - min1_pos) >= 5
        if not mask.any():
            return None
        
        filtered_lows = lows.iloc[mask]
        if len(filtered_lows) == 0:
            return None
        
        min2_idx = filtered_lows.idxmin()
        min2_val = filtered_lows[min2_idx]
        
        # 두 저점이 비슷한지 (tolerance 이내)
        diff_pct = abs(min1_val - min2_val) / min1_val * 100
        
        # 현재가가 두 저점보다 높은지 (반등)
        current_close = df['Close'].iloc[idx]
        above_both = current_close > max(min1_val, min2_val) * 1.02
        
        # 최근 상승 중인지
        recent_up = df['Close'].iloc[idx] > df['Close'].iloc[idx-3]
        
        if diff_pct <= self.tolerance_pct and above_both and recent_up:
            confidence = min(1.0, (self.tolerance_pct - diff_pct) / self.tolerance_pct)
            
            return PatternMatch(
                pattern_name=self.name,
                date_idx=idx,
                date=df.index[idx],
                confidence=confidence,
                details={
                    'bottom1': min1_val,
                    'bottom2': min2_val,
                    'diff_pct': diff_pct,
                    'current': current_close
                }
            )
        return None


class HigherLow(BasePattern):
    """저점 상승 패턴 (상승 추세 확인)"""
    
    name = "Higher_Low"
    description = "최근 저점이 이전 저점보다 높아지는 패턴"
    
    def __init__(self, period: int = 10):
        self.period = period
    
    def check(self, df: pd.DataFrame, idx: int) -> Optional[PatternMatch]:
        if idx < self.period * 3:
            return None
        
        # 3개 구간의 저점 비교
        low1 = df['Low'].iloc[idx-self.period*3:idx-self.period*2].min()
        low2 = df['Low'].iloc[idx-self.period*2:idx-self.period].min()
        low3 = df['Low'].iloc[idx-self.period:idx+1].min()
        
        # 조건: low1 < low2 < low3 (저점 상승)
        if low1 < low2 < low3:
            # 현재 가격이 최근 저점 근처에서 반등 중
            current_close = df['Close'].iloc[idx]
            near_low = (current_close - low3) / low3 * 100 < 5  # 저점 대비 5% 이내
            bouncing = df['Close'].iloc[idx] > df['Close'].iloc[idx-1]
            
            if near_low and bouncing:
                improvement = (low3 - low1) / low1 * 100
                confidence = min(1.0, improvement / 10)
                
                return PatternMatch(
                    pattern_name=self.name,
                    date_idx=idx,
                    date=df.index[idx],
                    confidence=confidence,
                    details={
                        'low1': low1,
                        'low2': low2,
                        'low3': low3,
                        'improvement_pct': improvement
                    }
                )
        return None


class InsideBarBreakout(BasePattern):
    """내부 봉 돌파 패턴"""
    
    name = "Inside_Bar_Breakout"
    description = "전일 범위 안에 있다가 상단 돌파"
    
    def check(self, df: pd.DataFrame, idx: int) -> Optional[PatternMatch]:
        if idx < 3:
            return None
        
        # 2일 전 (Mother bar)
        mother_high = df['High'].iloc[idx-2]
        mother_low = df['Low'].iloc[idx-2]
        
        # 어제 (Inside bar): Mother 범위 안에 있어야 함
        prev_high = df['High'].iloc[idx-1]
        prev_low = df['Low'].iloc[idx-1]
        
        is_inside = prev_high <= mother_high and prev_low >= mother_low
        
        if not is_inside:
            return None
        
        # 오늘: 상단 돌파
        curr_close = df['Close'].iloc[idx]
        breakout_up = curr_close > mother_high
        
        if breakout_up:
            breakout_strength = (curr_close - mother_high) / mother_high * 100
            confidence = min(1.0, breakout_strength * 2)
            
            return PatternMatch(
                pattern_name=self.name,
                date_idx=idx,
                date=df.index[idx],
                confidence=confidence,
                details={
                    'mother_high': mother_high,
                    'mother_low': mother_low,
                    'breakout_price': curr_close,
                    'breakout_pct': breakout_strength
                }
            )
        return None


class VolumeClimaxReversal(BasePattern):
    """거래량 클라이맥스 후 반전 패턴"""
    
    name = "Volume_Climax_Reversal"
    description = "극단적 거래량 + 하락 후 반등"
    
    def __init__(self, volume_threshold: float = 3.0, lookback: int = 20):
        self.volume_threshold = volume_threshold
        self.lookback = lookback
    
    def check(self, df: pd.DataFrame, idx: int) -> Optional[PatternMatch]:
        if idx < self.lookback + 5 or 'Volume' not in df.columns:
            return None
        
        # 최근 며칠 내 거래량 클라이맥스가 있었는지
        vol_avg = df['Volume'].iloc[idx-self.lookback:idx].mean()
        
        climax_found = False
        climax_idx = -1
        
        for i in range(idx-5, idx):
            vol_ratio = df['Volume'].iloc[i] / vol_avg
            price_drop = (df['Close'].iloc[i] / df['Close'].iloc[i-1] - 1) * 100
            
            # 거래량 3배 이상 + 가격 하락
            if vol_ratio >= self.volume_threshold and price_drop < -1:
                climax_found = True
                climax_idx = i
                break
        
        if not climax_found:
            return None
        
        # 클라이맥스 이후 반등 중인지
        climax_close = df['Close'].iloc[climax_idx]
        current_close = df['Close'].iloc[idx]
        recovery = (current_close / climax_close - 1) * 100
        
        if recovery > 1:  # 1% 이상 반등
            confidence = min(1.0, recovery / 5)
            
            return PatternMatch(
                pattern_name=self.name,
                date_idx=idx,
                date=df.index[idx],
                confidence=confidence,
                details={
                    'climax_date': str(df.index[climax_idx].date()),
                    'climax_volume_ratio': df['Volume'].iloc[climax_idx] / vol_avg,
                    'recovery_pct': recovery
                }
            )
        return None


# =============================================================================
# 시장 상태 필터
# =============================================================================

class MarketRegime:
    """시장 상태 분류"""
    
    BULL = "BULL"      # 상승장
    BEAR = "BEAR"      # 하락장
    SIDEWAYS = "SIDEWAYS"  # 횡보장
    
    @staticmethod
    def classify(df: pd.DataFrame, idx: int) -> str:
        """
        시장 상태 분류
        - 200일 MA 위 + 상승 추세 = BULL
        - 200일 MA 아래 + 하락 추세 = BEAR
        - 그 외 = SIDEWAYS
        """
        if idx < 200 or 'ma_long' not in df.columns:
            return MarketRegime.SIDEWAYS
        
        close = df['Close'].iloc[idx]
        ma_200 = df['ma_long'].iloc[idx]
        
        # 20일 추세 (최근 vs 20일 전)
        if idx >= 20:
            trend = (close / df['Close'].iloc[idx-20] - 1) * 100
        else:
            trend = 0
        
        if close > ma_200 and trend > 5:
            return MarketRegime.BULL
        elif close < ma_200 and trend < -5:
            return MarketRegime.BEAR
        else:
            return MarketRegime.SIDEWAYS
    
    @staticmethod
    def add_regime_column(df: pd.DataFrame) -> pd.DataFrame:
        """데이터프레임에 시장 상태 컬럼 추가"""
        df = df.copy()
        regimes = []
        for idx in range(len(df)):
            regimes.append(MarketRegime.classify(df, idx))
        df['market_regime'] = regimes
        return df


# =============================================================================
# 패턴 매니저 (앙상블)
# =============================================================================

class RuleBasedPatternManager:
    """규칙 기반 패턴 매니저"""
    
    def __init__(self):
        # 모든 패턴 등록 (15개)
        self.patterns = [
            # 기존 7개
            RSIOversold(),
            BollingerSqueeze(),
            GoldenCross(),
            VolumeSpike(),
            MACDCrossover(),
            PriceAtSupport(),
            MomentumReversal(),
            # 추가 8개
            StochasticOversold(),
            WilliamsROversold(),
            ATRBreakout(),
            BullishEngulfing(),
            DoubleBottom(),
            HigherLow(),
            InsideBarBreakout(),
            VolumeClimaxReversal()
        ]
    
    def scan_all(self, df: pd.DataFrame, 
                start_idx: int = 200,
                market_filter: str = None) -> Dict[str, List[PatternMatch]]:
        """
        모든 패턴 스캔
        
        Args:
            df: 지표가 계산된 데이터프레임
            start_idx: 시작 인덱스
            market_filter: 시장 상태 필터 (BULL/BEAR/SIDEWAYS/None)
        
        Returns:
            {패턴명: [매칭 리스트]}
        """
        # 시장 상태 추가
        df = MarketRegime.add_regime_column(df)
        
        results = {}
        for pattern in self.patterns:
            matches = []
            for idx in range(start_idx, len(df)):
                # 시장 필터 적용
                if market_filter and df['market_regime'].iloc[idx] != market_filter:
                    continue
                
                match = pattern.check(df, idx)
                if match:
                    matches.append(match)
            
            results[pattern.name] = matches
        
        return results
    
    def get_ensemble_signals(self, df: pd.DataFrame, idx: int,
                            min_patterns: int = 2) -> Optional[Dict]:
        """
        앙상블 신호 (여러 패턴 동시 발생)
        
        Args:
            df: 데이터프레임
            idx: 확인할 인덱스
            min_patterns: 최소 패턴 수
        
        Returns:
            앙상블 신호 정보 또는 None
        """
        triggered_patterns = []
        total_confidence = 0
        
        for pattern in self.patterns:
            match = pattern.check(df, idx)
            if match:
                triggered_patterns.append(match)
                total_confidence += match.confidence
        
        if len(triggered_patterns) >= min_patterns:
            return {
                'date_idx': idx,
                'date': df.index[idx],
                'n_patterns': len(triggered_patterns),
                'patterns': [m.pattern_name for m in triggered_patterns],
                'avg_confidence': total_confidence / len(triggered_patterns),
                'details': {m.pattern_name: m.details for m in triggered_patterns}
            }
        
        return None
    
    def scan_ensemble(self, df: pd.DataFrame,
                     start_idx: int = 200,
                     min_patterns: int = 2) -> List[Dict]:
        """앙상블 신호 전체 스캔"""
        signals = []
        for idx in range(start_idx, len(df)):
            signal = self.get_ensemble_signals(df, idx, min_patterns)
            if signal:
                signals.append(signal)
        return signals

