"""기술적 지표 계산 모듈"""

import pandas as pd
import numpy as np
from typing import Dict, Any


class TechnicalIndicators:
    """기술적 지표 계산 클래스"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Args:
            config: 지표 설정 (settings.yaml의 indicators 섹션)
        """
        self.config = config or {}
        
        # 기본값 설정
        self.rsi_period = self.config.get('rsi', {}).get('period', 14)
        self.macd_fast = self.config.get('macd', {}).get('fast', 12)
        self.macd_slow = self.config.get('macd', {}).get('slow', 26)
        self.macd_signal = self.config.get('macd', {}).get('signal', 9)
        self.bb_period = self.config.get('bollinger', {}).get('period', 20)
        self.bb_std = self.config.get('bollinger', {}).get('std', 2)
        self.ma_short = self.config.get('moving_averages', {}).get('short', 20)
        self.ma_medium = self.config.get('moving_averages', {}).get('medium', 50)
        self.ma_long = self.config.get('moving_averages', {}).get('long', 200)
        self.atr_period = self.config.get('atr', {}).get('period', 14)
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        모든 기술적 지표 계산
        
        Args:
            df: OHLCV 데이터프레임
        
        Returns:
            지표가 추가된 데이터프레임
        """
        df = df.copy()
        
        # 가격 변화율
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # 이동평균
        df['ma_short'] = df['Close'].rolling(window=self.ma_short).mean()
        df['ma_medium'] = df['Close'].rolling(window=self.ma_medium).mean()
        df['ma_long'] = df['Close'].rolling(window=self.ma_long).mean()
        
        # 이동평균 대비 위치 (%)
        df['price_vs_ma_short'] = (df['Close'] / df['ma_short'] - 1) * 100
        df['price_vs_ma_medium'] = (df['Close'] / df['ma_medium'] - 1) * 100
        df['price_vs_ma_long'] = (df['Close'] / df['ma_long'] - 1) * 100
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['Close'], self.rsi_period)
        
        # MACD
        macd_data = self._calculate_macd(df['Close'])
        df['macd'] = macd_data['macd']
        df['macd_signal'] = macd_data['signal']
        df['macd_hist'] = macd_data['histogram']
        
        # 볼린저 밴드
        bb_data = self._calculate_bollinger(df['Close'])
        df['bb_upper'] = bb_data['upper']
        df['bb_middle'] = bb_data['middle']
        df['bb_lower'] = bb_data['lower']
        df['bb_width'] = bb_data['width']
        df['bb_position'] = bb_data['position']  # 밴드 내 위치 (0~1)
        
        # ATR (변동성)
        df['atr'] = self._calculate_atr(df)
        df['atr_pct'] = df['atr'] / df['Close'] * 100  # 가격 대비 ATR %
        
        # 거래량 지표
        df['volume_ma'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma']  # 평균 대비 거래량
        
        # 변동성 지표
        df['volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252) * 100
        
        # 모멘텀 지표
        df['momentum_10'] = df['Close'].pct_change(periods=10) * 100
        df['momentum_20'] = df['Close'].pct_change(periods=20) * 100
        
        # 고가/저가 대비 위치
        df['high_20'] = df['High'].rolling(window=20).max()
        df['low_20'] = df['Low'].rolling(window=20).min()
        df['range_position'] = (df['Close'] - df['low_20']) / (df['high_20'] - df['low_20'])
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        
        # Wilder's smoothing
        for i in range(period, len(prices)):
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period-1) + gain.iloc[i]) / period
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period-1) + loss.iloc[i]) / period
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """MACD 계산"""
        ema_fast = prices.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = prices.ewm(span=self.macd_slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def _calculate_bollinger(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """볼린저 밴드 계산"""
        middle = prices.rolling(window=self.bb_period).mean()
        std = prices.rolling(window=self.bb_period).std()
        
        upper = middle + (self.bb_std * std)
        lower = middle - (self.bb_std * std)
        width = (upper - lower) / middle * 100  # 밴드폭 (%)
        
        # 밴드 내 위치 (0=하단, 1=상단)
        position = (prices - lower) / (upper - lower)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'width': width,
            'position': position
        }
    
    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """ATR (Average True Range) 계산"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()
        
        return atr

