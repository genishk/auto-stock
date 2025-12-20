"""
패턴 발견 모듈
- 수익 포인트에서 공통 패턴 추출
- 발생도 검증
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from .profit_cases import ProfitCaseFinder, ProfitCase


@dataclass
class PatternDefinition:
    """패턴 정의"""
    name: str
    category: str  # 카테고리 (모멘텀, 가격, 거래량 등)
    description: str
    conditions: Dict[str, Tuple[float, float]]  # {지표명: (min, max)}
    
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


@dataclass
class PatternStats:
    """패턴 통계"""
    name: str
    # 발견 기간 (앞 400개)
    discovery_count: int = 0
    discovery_total_days: int = 0
    discovery_frequency: float = 0.0
    
    # 검증 기간 (이후)
    validation_count: int = 0
    validation_total_days: int = 0
    validation_frequency: float = 0.0
    
    # 발생도 유지 비율
    frequency_ratio: float = 0.0  # 검증/발견 빈도 비율
    
    passed: bool = False  # 발생도 검증 통과 여부


class PatternFinder:
    """
    패턴 발견기
    
    1. 수익 포인트 직전의 공통 특징 분석
    2. 다양한 관점에서 패턴 정의
    3. 발생도 검증
    """
    
    def __init__(self, 
                 holding_period: int = 60,
                 min_return: float = 10.0,
                 discovery_ratio: float = 0.67):  # 앞 67% (400/600)
        """
        Args:
            holding_period: 보유 기간
            min_return: 최소 수익률
            discovery_ratio: 패턴 발견에 사용할 수익 케이스 비율
        """
        self.holding_period = holding_period
        self.min_return = min_return
        self.discovery_ratio = discovery_ratio
        
        self.profit_finder = ProfitCaseFinder(
            holding_periods=[holding_period],
            min_returns=[min_return]
        )
    
    def find_patterns(self, df: pd.DataFrame) -> Tuple[List[PatternDefinition], Dict]:
        """
        패턴 발견 메인 함수
        
        Args:
            df: 지표가 계산된 데이터프레임
        
        Returns:
            (패턴 정의 리스트, 분석 정보)
        """
        print("\n" + "="*70)
        print("🔍 패턴 발견 시작")
        print("="*70)
        
        # 1. 수익 케이스 찾기
        all_cases = self.profit_finder.find_profit_cases(df, self.holding_period, self.min_return)
        n_cases = len(all_cases)
        
        print(f"총 수익 케이스: {n_cases}개")
        print(f"발견용: 앞 {int(n_cases * self.discovery_ratio)}개")
        print(f"검증용 기간: 이후")
        
        # 2. 발견/검증 분할
        split_idx = int(n_cases * self.discovery_ratio)
        discovery_cases = all_cases[:split_idx]
        
        # 발견 기간의 마지막 날짜
        discovery_end_date = discovery_cases[-1].date
        discovery_end_idx = discovery_cases[-1].date_idx
        
        print(f"\n발견 기간: ~ {discovery_end_date.date()}")
        print(f"검증 기간: {discovery_end_date.date()} 이후")
        
        # 3. 수익 케이스 직전 특징 분석
        print(f"\n[1/3] 수익 케이스 직전 특징 분석...")
        feature_stats = self._analyze_features(df, discovery_cases)
        
        # 4. 패턴 정의
        print(f"\n[2/3] 패턴 정의 (다양한 관점)...")
        patterns = self._define_patterns(feature_stats, df, discovery_cases)
        
        print(f"   → {len(patterns)}개 패턴 정의됨")
        
        # 5. 발생도 검증
        print(f"\n[3/3] 발생도 검증...")
        validated_patterns, stats = self._validate_frequency(
            df, patterns, discovery_end_idx
        )
        
        info = {
            'total_cases': n_cases,
            'discovery_cases': len(discovery_cases),
            'discovery_end_date': str(discovery_end_date.date()),
            'patterns_defined': len(patterns),
            'patterns_passed': len(validated_patterns),
            'pattern_stats': stats
        }
        
        return validated_patterns, info
    
    def _analyze_features(self, df: pd.DataFrame, 
                         cases: List[ProfitCase]) -> Dict[str, Dict]:
        """수익 케이스 직전의 특징 분석"""
        
        # 분석할 지표들
        indicators = [
            'rsi', 'macd_hist', 'bb_position', 'momentum_10', 'momentum_20',
            'volume_ratio', 'volatility_20', 'returns', 'range_position',
            'price_vs_ma_short', 'price_vs_ma_medium', 'price_vs_ma_long'
        ]
        
        # 각 지표별 값 수집
        feature_values = defaultdict(list)
        
        for case in cases:
            idx = case.date_idx
            if idx < 1:
                continue
            
            row = df.iloc[idx]
            for ind in indicators:
                if ind in row.index and not pd.isna(row[ind]):
                    feature_values[ind].append(row[ind])
        
        # 통계 계산
        stats = {}
        for ind, values in feature_values.items():
            if len(values) < 10:
                continue
            
            arr = np.array(values)
            stats[ind] = {
                'mean': np.mean(arr),
                'std': np.std(arr),
                'min': np.min(arr),
                'max': np.max(arr),
                'p10': np.percentile(arr, 10),
                'p25': np.percentile(arr, 25),
                'p50': np.percentile(arr, 50),
                'p75': np.percentile(arr, 75),
                'p90': np.percentile(arr, 90),
                'count': len(values)
            }
        
        return stats
    
    def _define_patterns(self, feature_stats: Dict, 
                        df: pd.DataFrame,
                        cases: List[ProfitCase]) -> List[PatternDefinition]:
        """다양한 관점에서 패턴 정의"""
        
        patterns = []
        
        # =============================================
        # 1. RSI 기반 패턴 (5개)
        # =============================================
        if 'rsi' in feature_stats:
            s = feature_stats['rsi']
            
            # RSI 과매도
            patterns.append(PatternDefinition(
                name="RSI_Oversold_30",
                category="모멘텀",
                description="RSI 30 이하",
                conditions={'rsi': (0, 30)}
            ))
            
            patterns.append(PatternDefinition(
                name="RSI_Oversold_35",
                category="모멘텀",
                description="RSI 35 이하",
                conditions={'rsi': (0, 35)}
            ))
            
            patterns.append(PatternDefinition(
                name="RSI_Oversold_40",
                category="모멘텀",
                description="RSI 40 이하",
                conditions={'rsi': (0, 40)}
            ))
            
            # RSI 중립
            patterns.append(PatternDefinition(
                name="RSI_Neutral_Low",
                category="모멘텀",
                description="RSI 40-50",
                conditions={'rsi': (40, 50)}
            ))
            
            patterns.append(PatternDefinition(
                name="RSI_Neutral_High",
                category="모멘텀",
                description="RSI 50-60",
                conditions={'rsi': (50, 60)}
            ))
        
        # =============================================
        # 2. 모멘텀 기반 패턴 (6개)
        # =============================================
        if 'momentum_10' in feature_stats:
            # 하락 후 반등
            patterns.append(PatternDefinition(
                name="Momentum10_VeryNegative",
                category="모멘텀",
                description="10일 모멘텀 -10% 이하",
                conditions={'momentum_10': (-100, -10)}
            ))
            
            patterns.append(PatternDefinition(
                name="Momentum10_Negative",
                category="모멘텀",
                description="10일 모멘텀 -5% ~ -10%",
                conditions={'momentum_10': (-10, -5)}
            ))
            
            patterns.append(PatternDefinition(
                name="Momentum10_SlightNegative",
                category="모멘텀",
                description="10일 모멘텀 -5% ~ 0%",
                conditions={'momentum_10': (-5, 0)}
            ))
        
        if 'momentum_20' in feature_stats:
            patterns.append(PatternDefinition(
                name="Momentum20_VeryNegative",
                category="모멘텀",
                description="20일 모멘텀 -15% 이하",
                conditions={'momentum_20': (-100, -15)}
            ))
            
            patterns.append(PatternDefinition(
                name="Momentum20_Negative",
                category="모멘텀",
                description="20일 모멘텀 -10% ~ -15%",
                conditions={'momentum_20': (-15, -10)}
            ))
            
            patterns.append(PatternDefinition(
                name="Momentum20_SlightNegative",
                category="모멘텀",
                description="20일 모멘텀 -10% ~ 0%",
                conditions={'momentum_20': (-10, 0)}
            ))
        
        # =============================================
        # 3. 볼린저 밴드 기반 패턴 (4개)
        # =============================================
        if 'bb_position' in feature_stats:
            patterns.append(PatternDefinition(
                name="BB_BelowLower",
                category="변동성",
                description="볼린저밴드 하단 돌파 (position < 0)",
                conditions={'bb_position': (-10, 0)}
            ))
            
            patterns.append(PatternDefinition(
                name="BB_NearLower",
                category="변동성",
                description="볼린저밴드 하단 근처 (0-0.2)",
                conditions={'bb_position': (0, 0.2)}
            ))
            
            patterns.append(PatternDefinition(
                name="BB_LowerHalf",
                category="변동성",
                description="볼린저밴드 하단 절반 (0.2-0.5)",
                conditions={'bb_position': (0.2, 0.5)}
            ))
            
            patterns.append(PatternDefinition(
                name="BB_Middle",
                category="변동성",
                description="볼린저밴드 중간 (0.4-0.6)",
                conditions={'bb_position': (0.4, 0.6)}
            ))
        
        # =============================================
        # 4. 이동평균 대비 가격 (6개)
        # =============================================
        if 'price_vs_ma_short' in feature_stats:
            patterns.append(PatternDefinition(
                name="Price_Below_MA20_5pct",
                category="추세",
                description="20일 MA 대비 -5% 이상 하락",
                conditions={'price_vs_ma_short': (-100, -5)}
            ))
            
            patterns.append(PatternDefinition(
                name="Price_Below_MA20_2pct",
                category="추세",
                description="20일 MA 대비 -2% ~ -5%",
                conditions={'price_vs_ma_short': (-5, -2)}
            ))
            
            patterns.append(PatternDefinition(
                name="Price_Near_MA20",
                category="추세",
                description="20일 MA 근처 (-2% ~ +2%)",
                conditions={'price_vs_ma_short': (-2, 2)}
            ))
        
        if 'price_vs_ma_medium' in feature_stats:
            patterns.append(PatternDefinition(
                name="Price_Below_MA50_10pct",
                category="추세",
                description="50일 MA 대비 -10% 이상 하락",
                conditions={'price_vs_ma_medium': (-100, -10)}
            ))
            
            patterns.append(PatternDefinition(
                name="Price_Below_MA50_5pct",
                category="추세",
                description="50일 MA 대비 -5% ~ -10%",
                conditions={'price_vs_ma_medium': (-10, -5)}
            ))
            
            patterns.append(PatternDefinition(
                name="Price_Near_MA50",
                category="추세",
                description="50일 MA 근처 (-5% ~ +2%)",
                conditions={'price_vs_ma_medium': (-5, 2)}
            ))
        
        # =============================================
        # 5. 거래량 기반 패턴 (3개)
        # =============================================
        if 'volume_ratio' in feature_stats:
            patterns.append(PatternDefinition(
                name="Volume_Spike",
                category="거래량",
                description="거래량 2배 이상 급증",
                conditions={'volume_ratio': (2, 100)}
            ))
            
            patterns.append(PatternDefinition(
                name="Volume_High",
                category="거래량",
                description="거래량 1.5배 이상",
                conditions={'volume_ratio': (1.5, 2)}
            ))
            
            patterns.append(PatternDefinition(
                name="Volume_Normal",
                category="거래량",
                description="거래량 0.8-1.2배",
                conditions={'volume_ratio': (0.8, 1.2)}
            ))
        
        # =============================================
        # 6. 변동성 기반 패턴 (3개)
        # =============================================
        if 'volatility_20' in feature_stats:
            s = feature_stats['volatility_20']
            
            patterns.append(PatternDefinition(
                name="Volatility_High",
                category="변동성",
                description=f"높은 변동성 (상위 25%)",
                conditions={'volatility_20': (s['p75'], s['max'] * 2)}
            ))
            
            patterns.append(PatternDefinition(
                name="Volatility_Medium",
                category="변동성",
                description=f"중간 변동성",
                conditions={'volatility_20': (s['p25'], s['p75'])}
            ))
            
            patterns.append(PatternDefinition(
                name="Volatility_Low",
                category="변동성",
                description=f"낮은 변동성 (하위 25%)",
                conditions={'volatility_20': (0, s['p25'])}
            ))
        
        # =============================================
        # 7. 복합 패턴 (6개) - 여러 조건 조합
        # =============================================
        
        # RSI 과매도 + 하락 모멘텀
        patterns.append(PatternDefinition(
            name="Combo_Oversold_Momentum",
            category="복합",
            description="RSI < 40 AND 10일 모멘텀 < -5%",
            conditions={
                'rsi': (0, 40),
                'momentum_10': (-100, -5)
            }
        ))
        
        # 볼린저 하단 + RSI 과매도
        patterns.append(PatternDefinition(
            name="Combo_BB_RSI_Oversold",
            category="복합",
            description="BB 하단 근처 AND RSI < 40",
            conditions={
                'bb_position': (-10, 0.3),
                'rsi': (0, 40)
            }
        ))
        
        # MA 하회 + 거래량 증가
        patterns.append(PatternDefinition(
            name="Combo_Below_MA_Volume",
            category="복합",
            description="20일 MA -3% 이상 하회 AND 거래량 1.3배+",
            conditions={
                'price_vs_ma_short': (-100, -3),
                'volume_ratio': (1.3, 100)
            }
        ))
        
        # 강한 하락 + 반등 조짐
        patterns.append(PatternDefinition(
            name="Combo_Strong_Dip",
            category="복합",
            description="20일 모멘텀 < -10% AND BB 하단",
            conditions={
                'momentum_20': (-100, -10),
                'bb_position': (-10, 0.3)
            }
        ))
        
        # 적당한 조정 + 중립 RSI
        patterns.append(PatternDefinition(
            name="Combo_Mild_Pullback",
            category="복합",
            description="10일 모멘텀 -3%~0% AND RSI 40-55",
            conditions={
                'momentum_10': (-3, 0),
                'rsi': (40, 55)
            }
        ))
        
        # 깊은 조정 + 높은 변동성
        if 'volatility_20' in feature_stats:
            s = feature_stats['volatility_20']
            patterns.append(PatternDefinition(
                name="Combo_Deep_Dip_HighVol",
                category="복합",
                description="20일 모멘텀 < -15% AND 높은 변동성",
                conditions={
                    'momentum_20': (-100, -15),
                    'volatility_20': (s['p50'], s['max'] * 2)
                }
            ))
        
        return patterns
    
    def _validate_frequency(self, df: pd.DataFrame,
                           patterns: List[PatternDefinition],
                           discovery_end_idx: int) -> Tuple[List[PatternDefinition], List[PatternStats]]:
        """발생도 검증"""
        
        # 발견 기간 / 검증 기간 분리
        df_discovery = df.iloc[:discovery_end_idx + 1]
        df_validation = df.iloc[discovery_end_idx + 1:]
        
        stats_list = []
        passed_patterns = []
        
        for pattern in patterns:
            # 발견 기간 발생 횟수
            discovery_count = 0
            for idx in range(len(df_discovery)):
                if pattern.check(df_discovery.iloc[idx]):
                    discovery_count += 1
            
            # 검증 기간 발생 횟수
            validation_count = 0
            for idx in range(len(df_validation)):
                if pattern.check(df_validation.iloc[idx]):
                    validation_count += 1
            
            # 빈도 계산
            discovery_freq = discovery_count / len(df_discovery) * 100 if len(df_discovery) > 0 else 0
            validation_freq = validation_count / len(df_validation) * 100 if len(df_validation) > 0 else 0
            
            # 빈도 비율 (검증/발견)
            freq_ratio = validation_freq / discovery_freq if discovery_freq > 0 else 0
            
            # 통과 기준:
            # 1. 발견 기간에 최소 10회 이상 발생
            # 2. 검증 기간에도 최소 5회 이상 발생
            # 3. 빈도 비율 0.3 이상 (너무 많이 줄어들면 안 됨)
            passed = (
                discovery_count >= 10 and
                validation_count >= 5 and
                freq_ratio >= 0.3
            )
            
            stats = PatternStats(
                name=pattern.name,
                discovery_count=discovery_count,
                discovery_total_days=len(df_discovery),
                discovery_frequency=discovery_freq,
                validation_count=validation_count,
                validation_total_days=len(df_validation),
                validation_frequency=validation_freq,
                frequency_ratio=freq_ratio,
                passed=passed
            )
            
            stats_list.append(stats)
            
            if passed:
                passed_patterns.append(pattern)
        
        # 결과 출력
        print(f"\n{'─'*80}")
        print(f"{'패턴명':<35} {'발견':^15} {'검증':^15} {'비율':^8} {'통과'}")
        print(f"{'─'*80}")
        
        for s in sorted(stats_list, key=lambda x: x.frequency_ratio, reverse=True):
            disc_str = f"{s.discovery_count}회 ({s.discovery_frequency:.1f}%)"
            val_str = f"{s.validation_count}회 ({s.validation_frequency:.1f}%)"
            ratio_str = f"{s.frequency_ratio:.2f}"
            passed_str = "✅" if s.passed else "❌"
            
            print(f"{s.name:<35} {disc_str:^15} {val_str:^15} {ratio_str:^8} {passed_str}")
        
        print(f"{'─'*80}")
        print(f"통과: {len(passed_patterns)}/{len(patterns)} 패턴")
        
        return passed_patterns, stats_list


@dataclass
class PatternPerformance:
    """패턴 수익률 검증 결과"""
    name: str
    
    # Train 기간
    train_pattern_days: int = 0      # 패턴 발생 일수
    train_profit_days: int = 0       # 그 중 수익 케이스인 일수
    train_win_rate: float = 0.0      # 승률 (수익/발생)
    train_avg_return: float = 0.0    # 평균 수익률
    
    # Test 기간
    test_pattern_days: int = 0
    test_profit_days: int = 0
    test_win_rate: float = 0.0
    test_avg_return: float = 0.0
    
    # 기준선 대비
    baseline_win_rate: float = 0.0   # 랜덤 확률
    lift_train: float = 0.0          # Train 승률 / 랜덤 확률
    lift_test: float = 0.0           # Test 승률 / 랜덤 확률
    
    validated: bool = False


class PatternValidator:
    """
    패턴 수익률 검증기
    
    패턴 발생 시점에 매수하면 실제로 수익이 나는지 검증
    """
    
    def __init__(self,
                 holding_period: int = 60,
                 min_return: float = 10.0,
                 train_ratio: float = 0.7):
        self.holding_period = holding_period
        self.min_return = min_return
        self.train_ratio = train_ratio
    
    def validate_patterns(self, df: pd.DataFrame,
                         patterns: List[PatternDefinition]) -> Tuple[List[PatternDefinition], List[PatternPerformance]]:
        """
        패턴 수익률 검증
        
        Args:
            df: 지표가 계산된 데이터프레임
            patterns: 검증할 패턴 리스트
        
        Returns:
            (검증 통과 패턴, 성과 리스트)
        """
        print("\n" + "="*70)
        print("📈 패턴 수익률 검증 (2차)")
        print("="*70)
        
        # Train/Test 분할
        split_idx = int(len(df) * self.train_ratio)
        df_train = df.iloc[:split_idx]
        df_test = df.iloc[split_idx:]
        
        print(f"Train: {df_train.index[0].date()} ~ {df_train.index[-1].date()} ({len(df_train)}일)")
        print(f"Test:  {df_test.index[0].date()} ~ {df_test.index[-1].date()} ({len(df_test)}일)")
        
        # 기준선 (랜덤 확률) 계산
        baseline_train = self._calculate_baseline(df_train)
        baseline_test = self._calculate_baseline(df_test)
        
        print(f"\n기준선 (랜덤 확률):")
        print(f"  Train: {baseline_train*100:.1f}%")
        print(f"  Test:  {baseline_test*100:.1f}%")
        
        # 각 패턴 검증
        performances = []
        validated_patterns = []
        
        for pattern in patterns:
            perf = self._validate_single_pattern(
                df_train, df_test, pattern, baseline_train, baseline_test
            )
            performances.append(perf)
            
            if perf.validated:
                validated_patterns.append(pattern)
        
        # 결과 출력
        self._print_results(performances, baseline_train, baseline_test)
        
        return validated_patterns, performances
    
    def _calculate_baseline(self, df: pd.DataFrame) -> float:
        """기준선 (랜덤 확률) 계산"""
        profit_count = 0
        total_count = 0
        
        for idx in range(len(df) - self.holding_period):
            entry = df['Close'].iloc[idx]
            exit_price = df['Close'].iloc[idx + self.holding_period]
            ret = (exit_price / entry - 1) * 100
            
            total_count += 1
            if ret >= self.min_return:
                profit_count += 1
        
        return profit_count / total_count if total_count > 0 else 0
    
    def _validate_single_pattern(self, df_train: pd.DataFrame,
                                df_test: pd.DataFrame,
                                pattern: PatternDefinition,
                                baseline_train: float,
                                baseline_test: float) -> PatternPerformance:
        """단일 패턴 검증"""
        
        # Train 기간 검증
        train_stats = self._check_pattern_returns(df_train, pattern)
        
        # Test 기간 검증
        test_stats = self._check_pattern_returns(df_test, pattern)
        
        # Lift 계산 (승률 / 랜덤 확률)
        lift_train = train_stats['win_rate'] / baseline_train if baseline_train > 0 else 0
        lift_test = test_stats['win_rate'] / baseline_test if baseline_test > 0 else 0
        
        # 검증 통과 기준:
        # 1. Train에서 최소 20회 이상 발생
        # 2. Test에서 최소 10회 이상 발생
        # 3. Train 승률 > 랜덤 + 5%p
        # 4. Test 승률 > 랜덤 + 5%p
        # 5. Train과 Test 모두 Lift > 1.2 (랜덤보다 20% 이상 좋아야)
        validated = (
            train_stats['pattern_days'] >= 20 and
            test_stats['pattern_days'] >= 10 and
            train_stats['win_rate'] > baseline_train + 0.05 and
            test_stats['win_rate'] > baseline_test + 0.05 and
            lift_train >= 1.2 and
            lift_test >= 1.2
        )
        
        return PatternPerformance(
            name=pattern.name,
            train_pattern_days=train_stats['pattern_days'],
            train_profit_days=train_stats['profit_days'],
            train_win_rate=train_stats['win_rate'],
            train_avg_return=train_stats['avg_return'],
            test_pattern_days=test_stats['pattern_days'],
            test_profit_days=test_stats['profit_days'],
            test_win_rate=test_stats['win_rate'],
            test_avg_return=test_stats['avg_return'],
            baseline_win_rate=baseline_train,
            lift_train=lift_train,
            lift_test=lift_test,
            validated=validated
        )
    
    def _check_pattern_returns(self, df: pd.DataFrame,
                              pattern: PatternDefinition) -> Dict:
        """패턴 발생 시 수익률 체크"""
        pattern_days = 0
        profit_days = 0
        returns = []
        
        for idx in range(len(df) - self.holding_period):
            row = df.iloc[idx]
            
            # 패턴 매칭 확인
            if not pattern.check(row):
                continue
            
            pattern_days += 1
            
            # 수익률 계산
            entry = df['Close'].iloc[idx]
            exit_price = df['Close'].iloc[idx + self.holding_period]
            ret = (exit_price / entry - 1) * 100
            returns.append(ret)
            
            if ret >= self.min_return:
                profit_days += 1
        
        win_rate = profit_days / pattern_days if pattern_days > 0 else 0
        avg_return = np.mean(returns) if returns else 0
        
        return {
            'pattern_days': pattern_days,
            'profit_days': profit_days,
            'win_rate': win_rate,
            'avg_return': avg_return
        }
    
    def _print_results(self, performances: List[PatternPerformance],
                      baseline_train: float, baseline_test: float):
        """결과 출력"""
        print(f"\n{'─'*100}")
        print(f"{'패턴명':<30} {'Train':^30} {'Test':^30} {'Lift':^12} {'통과'}")
        print(f"{'':<30} {'발생 → 승률':^30} {'발생 → 승률':^30} {'Tr   Te':^12}")
        print(f"{'─'*100}")
        
        # 정렬 (Test 승률 기준)
        sorted_perfs = sorted(performances, key=lambda x: x.test_win_rate, reverse=True)
        
        for p in sorted_perfs:
            train_str = f"{p.train_pattern_days}회 → {p.train_win_rate*100:.1f}% (avg {p.train_avg_return:.1f}%)"
            test_str = f"{p.test_pattern_days}회 → {p.test_win_rate*100:.1f}% (avg {p.test_avg_return:.1f}%)"
            lift_str = f"{p.lift_train:.2f} {p.lift_test:.2f}"
            passed_str = "✅" if p.validated else "❌"
            
            print(f"{p.name:<30} {train_str:^30} {test_str:^30} {lift_str:^12} {passed_str}")
        
        print(f"{'─'*100}")
        
        validated_count = sum(1 for p in performances if p.validated)
        print(f"\n기준선: Train {baseline_train*100:.1f}%, Test {baseline_test*100:.1f}%")
        print(f"검증 통과: {validated_count}/{len(performances)} 패턴")
        
        if validated_count > 0:
            print(f"\n✅ 검증 통과 패턴:")
            for p in sorted_perfs:
                if p.validated:
                    print(f"  - {p.name}: Test 승률 {p.test_win_rate*100:.1f}% (Lift {p.lift_test:.2f}x)")


def run_pattern_discovery(df: pd.DataFrame,
                         holding_period: int = 60,
                         min_return: float = 10.0) -> Tuple[List[PatternDefinition], Dict]:
    """
    패턴 발견 실행 함수
    
    Args:
        df: 지표가 계산된 데이터프레임
        holding_period: 보유 기간
        min_return: 최소 수익률
    
    Returns:
        (통과한 패턴 리스트, 분석 정보)
    """
    finder = PatternFinder(
        holding_period=holding_period,
        min_return=min_return
    )
    
    return finder.find_patterns(df)


def run_full_pipeline(df: pd.DataFrame,
                     holding_period: int = 60,
                     min_return: float = 10.0) -> Tuple[List[PatternDefinition], Dict]:
    """
    전체 파이프라인 실행 (패턴 발견 + 수익률 검증)
    
    Args:
        df: 지표가 계산된 데이터프레임
        holding_period: 보유 기간
        min_return: 최소 수익률
    
    Returns:
        (최종 검증된 패턴, 전체 정보)
    """
    # 1. 패턴 발견 + 발생도 검증
    patterns_freq_validated, discovery_info = run_pattern_discovery(
        df, holding_period, min_return
    )
    
    if not patterns_freq_validated:
        print("\n⚠️ 발생도 검증 통과 패턴 없음")
        return [], discovery_info
    
    # 2. 수익률 검증
    validator = PatternValidator(
        holding_period=holding_period,
        min_return=min_return
    )
    
    final_patterns, performances = validator.validate_patterns(df, patterns_freq_validated)
    
    # 정보 병합
    discovery_info['profit_validation'] = {
        'input_patterns': len(patterns_freq_validated),
        'validated_patterns': len(final_patterns),
        'performances': [
            {
                'name': p.name,
                'train_win_rate': p.train_win_rate,
                'test_win_rate': p.test_win_rate,
                'lift_test': p.lift_test,
                'validated': p.validated
            }
            for p in performances
        ]
    }
    
    return final_patterns, discovery_info

