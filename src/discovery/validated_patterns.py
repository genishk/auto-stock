"""
검증된 패턴 정의
- 발생도 검증 + 수익률 검증 모두 통과한 14개 패턴
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class ValidatedPattern:
    """검증된 패턴"""
    name: str
    category: str
    description: str
    conditions: Dict[str, tuple]  # {지표명: (min, max)}
    
    # 검증 결과
    train_occurrences: int
    train_win_rate: float
    train_avg_return: float
    test_occurrences: int
    test_win_rate: float
    test_avg_return: float
    lift: float  # Test 승률 / 기준선
    
    def check(self, row: pd.Series) -> bool:
        """해당 row가 패턴 조건을 만족하는지"""
        for indicator, (min_val, max_val) in self.conditions.items():
            if indicator not in row.index:
                return False
            val = row[indicator]
            if pd.isna(val):
                return False
            if not (min_val <= val <= max_val):
                return False
        return True
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'category': self.category,
            'description': self.description,
            'conditions': self.conditions,
            'train_occurrences': self.train_occurrences,
            'train_win_rate': self.train_win_rate,
            'train_avg_return': self.train_avg_return,
            'test_occurrences': self.test_occurrences,
            'test_win_rate': self.test_win_rate,
            'test_avg_return': self.test_avg_return,
            'lift': self.lift
        }


# 검증된 14개 패턴 (2차 검증 통과)
VALIDATED_PATTERNS = [
    # 1. Combo_Strong_Dip - 최고 성과
    ValidatedPattern(
        name="Combo_Strong_Dip",
        category="복합",
        description="20일 모멘텀 < -10% AND BB 하단",
        conditions={
            'momentum_20': (-100, -10),
            'bb_position': (-10, 0.3)
        },
        train_occurrences=46,
        train_win_rate=0.370,
        train_avg_return=9.5,
        test_occurrences=11,
        test_win_rate=1.000,
        test_avg_return=17.6,
        lift=2.84
    ),
    
    # 2. Momentum20_Negative
    ValidatedPattern(
        name="Momentum20_Negative",
        category="모멘텀",
        description="20일 모멘텀 -10% ~ -15%",
        conditions={'momentum_20': (-15, -10)},
        train_occurrences=41,
        train_win_rate=0.268,
        train_avg_return=5.7,
        test_occurrences=13,
        test_win_rate=0.923,
        test_avg_return=16.6,
        lift=2.62
    ),
    
    # 3. RSI_Oversold_35
    ValidatedPattern(
        name="RSI_Oversold_35",
        category="모멘텀",
        description="RSI 35 이하",
        conditions={'rsi': (0, 35)},
        train_occurrences=68,
        train_win_rate=0.368,
        train_avg_return=7.3,
        test_occurrences=23,
        test_win_rate=0.739,
        test_avg_return=15.2,
        lift=2.10
    ),
    
    # 4. BB_BelowLower
    ValidatedPattern(
        name="BB_BelowLower",
        category="변동성",
        description="볼린저밴드 하단 돌파 (position < 0)",
        conditions={'bb_position': (-10, 0)},
        train_occurrences=71,
        train_win_rate=0.352,
        train_avg_return=7.7,
        test_occurrences=23,
        test_win_rate=0.739,
        test_avg_return=14.7,
        lift=2.10
    ),
    
    # 5. Price_Below_MA20_5pct
    ValidatedPattern(
        name="Price_Below_MA20_5pct",
        category="추세",
        description="20일 MA 대비 -5% 이상 하락",
        conditions={'price_vs_ma_short': (-100, -5)},
        train_occurrences=93,
        train_win_rate=0.430,
        train_avg_return=8.9,
        test_occurrences=23,
        test_win_rate=0.739,
        test_avg_return=15.3,
        lift=2.10
    ),
    
    # 6. RSI_Oversold_40
    ValidatedPattern(
        name="RSI_Oversold_40",
        category="모멘텀",
        description="RSI 40 이하",
        conditions={'rsi': (0, 40)},
        train_occurrences=177,
        train_win_rate=0.305,
        train_avg_return=6.3,
        test_occurrences=60,
        test_win_rate=0.733,
        test_avg_return=13.8,
        lift=2.08
    ),
    
    # 7. Combo_BB_RSI_Oversold
    ValidatedPattern(
        name="Combo_BB_RSI_Oversold",
        category="복합",
        description="BB 하단 근처 AND RSI < 40",
        conditions={
            'bb_position': (-10, 0.3),
            'rsi': (0, 40)
        },
        train_occurrences=171,
        train_win_rate=0.304,
        train_avg_return=6.3,
        test_occurrences=59,
        test_win_rate=0.729,
        test_avg_return=13.9,
        lift=2.07
    ),
    
    # 8. Momentum10_Negative
    ValidatedPattern(
        name="Momentum10_Negative",
        category="모멘텀",
        description="10일 모멘텀 -5% ~ -10%",
        conditions={'momentum_10': (-10, -5)},
        train_occurrences=134,
        train_win_rate=0.313,
        train_avg_return=5.0,
        test_occurrences=41,
        test_win_rate=0.683,
        test_avg_return=13.1,
        lift=1.94
    ),
    
    # 9. Combo_Oversold_Momentum
    ValidatedPattern(
        name="Combo_Oversold_Momentum",
        category="복합",
        description="RSI < 40 AND 10일 모멘텀 < -5%",
        conditions={
            'rsi': (0, 40),
            'momentum_10': (-100, -5)
        },
        train_occurrences=115,
        train_win_rate=0.374,
        train_avg_return=8.0,
        test_occurrences=35,
        test_win_rate=0.657,
        test_avg_return=13.8,
        lift=1.86
    ),
    
    # 10. Price_Below_MA20_2pct
    ValidatedPattern(
        name="Price_Below_MA20_2pct",
        category="추세",
        description="20일 MA 대비 -2% ~ -5%",
        conditions={'price_vs_ma_short': (-5, -2)},
        train_occurrences=194,
        train_win_rate=0.268,
        train_avg_return=4.0,
        test_occurrences=87,
        test_win_rate=0.644,
        test_avg_return=11.6,
        lift=1.83
    ),
    
    # 11. Combo_Below_MA_Volume
    ValidatedPattern(
        name="Combo_Below_MA_Volume",
        category="복합",
        description="20일 MA -3% 이상 하회 AND 거래량 1.3배+",
        conditions={
            'price_vs_ma_short': (-100, -3),
            'volume_ratio': (1.3, 100)
        },
        train_occurrences=96,
        train_win_rate=0.365,
        train_avg_return=6.0,
        test_occurrences=25,
        test_win_rate=0.600,
        test_avg_return=13.6,
        lift=1.70
    ),
    
    # 12. BB_NearLower
    ValidatedPattern(
        name="BB_NearLower",
        category="변동성",
        description="볼린저밴드 하단 근처 (0-0.2)",
        conditions={'bb_position': (0, 0.2)},
        train_occurrences=160,
        train_win_rate=0.300,
        train_avg_return=5.5,
        test_occurrences=60,
        test_win_rate=0.567,
        test_avg_return=9.6,
        lift=1.61
    ),
    
    # 13. Momentum20_SlightNegative
    ValidatedPattern(
        name="Momentum20_SlightNegative",
        category="모멘텀",
        description="20일 모멘텀 -10% ~ 0%",
        conditions={'momentum_20': (-10, 0)},
        train_occurrences=487,
        train_win_rate=0.269,
        train_avg_return=4.6,
        test_occurrences=178,
        test_win_rate=0.556,
        test_avg_return=10.5,
        lift=1.58
    ),
    
    # 14. RSI_Neutral_Low
    ValidatedPattern(
        name="RSI_Neutral_Low",
        category="모멘텀",
        description="RSI 40-50",
        conditions={'rsi': (40, 50)},
        train_occurrences=331,
        train_win_rate=0.293,
        train_avg_return=4.6,
        test_occurrences=116,
        test_win_rate=0.509,
        test_avg_return=9.2,
        lift=1.44
    ),
]


def get_validated_patterns() -> List[ValidatedPattern]:
    """검증된 패턴 리스트 반환"""
    return VALIDATED_PATTERNS


def get_pattern_by_name(name: str) -> Optional[ValidatedPattern]:
    """이름으로 패턴 찾기"""
    for p in VALIDATED_PATTERNS:
        if p.name == name:
            return p
    return None


def check_signals(df: pd.DataFrame, lookback_days: int = 7) -> List[Dict]:
    """
    최근 N일간 발생한 신호 확인
    
    Args:
        df: 지표가 계산된 데이터프레임
        lookback_days: 확인할 최근 일수
    
    Returns:
        신호 리스트
    """
    signals = []
    
    end_idx = len(df) - 1
    start_idx = max(0, end_idx - lookback_days + 1)
    
    for idx in range(start_idx, end_idx + 1):
        row = df.iloc[idx]
        price = row['Close']
        date = df.index[idx]
        days_ago = end_idx - idx
        
        for pattern in VALIDATED_PATTERNS:
            if pattern.check(row):
                signals.append({
                    'pattern': pattern.name,
                    'category': pattern.category,
                    'description': pattern.description,
                    'date': date,
                    'days_ago': days_ago,
                    'price': price,
                    'test_win_rate': pattern.test_win_rate,
                    'test_avg_return': pattern.test_avg_return,
                    'lift': pattern.lift
                })
    
    # 최신순 정렬, 같은 날이면 승률 높은 순
    signals.sort(key=lambda x: (x['days_ago'], -x['test_win_rate']))
    
    return signals

