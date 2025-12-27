"""
QQQ 전략 파라미터 최적화
- 동일 금액($1,000) 기준 계산
- 10년 데이터 사용
- 576개 조합 테스트
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm

from src.data.cache import DataCache
from src.data.fetcher import DataFetcher
from src.data.validator import DataValidator
from src.features.technical import TechnicalIndicators
from src.utils.helpers import load_config

# 상수
CAPITAL_PER_ENTRY = 1000


def load_data(ticker='QQQ'):
    """데이터 로드"""
    config = load_config()
    cache = DataCache(
        cache_dir=str(Path(__file__).parent / config['data']['cache']['directory']),
        max_age_hours=24
    )
    
    df = cache.get(ticker)
    if df is None:
        fetcher = DataFetcher([ticker])
        data = fetcher.fetch(config['data']['period'])
        if ticker in data:
            df = data[ticker]
            df, _ = DataValidator.validate(df, ticker)
            cache.set(ticker, df)
    
    if df is not None:
        indicators = TechnicalIndicators(config.get('indicators', {}))
        df = indicators.calculate_all(df)
    
    return df


def add_ma_indicators(df, ma_short, ma_long):
    """이동평균선 추가"""
    df = df.copy()
    df['MA_short'] = df['Close'].rolling(window=ma_short).mean()
    df['MA_long'] = df['Close'].rolling(window=ma_long).mean()
    df['golden_cross'] = df['MA_short'] > df['MA_long']
    return df


def find_buy_signals(df, rsi_oversold, rsi_exit, use_golden_cross):
    """매수 시그널 찾기"""
    buy_signals = []
    in_oversold = False
    last_signal_date = None
    last_signal_price = None
    
    for idx in range(len(df)):
        rsi = df['rsi'].iloc[idx]
        if pd.isna(rsi):
            continue
        
        golden_cross_ok = True
        if use_golden_cross and 'golden_cross' in df.columns:
            gc = df['golden_cross'].iloc[idx]
            golden_cross_ok = gc if not pd.isna(gc) else False
        
        if rsi < rsi_oversold:
            in_oversold = True
            last_signal_date = df.index[idx]
            last_signal_price = df['Close'].iloc[idx]
        else:
            if in_oversold and rsi >= rsi_exit and last_signal_date is not None:
                if golden_cross_ok:
                    buy_signals.append({
                        'signal_date': last_signal_date,
                        'confirm_date': df.index[idx],
                        'confirm_price': df['Close'].iloc[idx]
                    })
                    in_oversold = False
                    last_signal_date = None
    
    return buy_signals


def find_sell_signals(df, rsi_overbought, rsi_exit):
    """매도 시그널 찾기"""
    sell_signals = []
    in_overbought = False
    last_signal_date = None
    last_signal_price = None
    
    for idx in range(len(df)):
        rsi = df['rsi'].iloc[idx]
        if pd.isna(rsi):
            continue
        
        if rsi > rsi_overbought:
            in_overbought = True
            last_signal_date = df.index[idx]
            last_signal_price = df['Close'].iloc[idx]
        else:
            if in_overbought and rsi <= rsi_exit and last_signal_date is not None:
                sell_signals.append({
                    'signal_date': last_signal_date,
                    'confirm_date': df.index[idx],
                    'confirm_price': df['Close'].iloc[idx]
                })
                in_overbought = False
                last_signal_date = None
    
    return sell_signals


def simulate_trades(df, buy_signals, sell_signals):
    """
    거래 시뮬레이션 (동일 금액 기준, profit_only)
    """
    all_buy_dates = {bs['confirm_date']: bs for bs in buy_signals}
    all_sell_dates = {ss['confirm_date']: ss for ss in sell_signals}
    
    trades = []
    positions = []
    
    for idx in range(len(df)):
        current_date = df.index[idx]
        current_price = df['Close'].iloc[idx]
        
        if positions:
            # 동일 금액 기준 평균가 계산
            total_invested = len(positions) * CAPITAL_PER_ENTRY
            total_quantity = sum(CAPITAL_PER_ENTRY / p['price'] for p in positions)
            avg_price = total_invested / total_quantity
            current_return = (current_price / avg_price - 1) * 100
            
            exit_reason = None
            exit_price = current_price
            
            # profit_only: 수익일 때만 매도
            if current_date in all_sell_dates:
                sell_price = all_sell_dates[current_date]['confirm_price']
                sell_return = (sell_price / avg_price - 1) * 100
                if sell_return > 0:
                    exit_reason = "익절"
                    exit_price = sell_price
            
            if exit_reason:
                final_return = (exit_price / avg_price - 1) * 100
                trades.append({
                    'entry_dates': [p['date'] for p in positions],
                    'entry_prices': [p['price'] for p in positions],
                    'avg_price': avg_price,
                    'num_buys': len(positions),
                    'exit_date': current_date,
                    'exit_price': exit_price,
                    'return': final_return,
                    'exit_reason': exit_reason
                })
                positions = []
        
        if current_date in all_buy_dates:
            positions.append({
                'date': current_date,
                'price': all_buy_dates[current_date]['confirm_price']
            })
    
    return trades, positions


def calculate_performance(trades):
    """성과 계산 (동일 금액 기준)"""
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'total_invested': 0,
            'total_profit': 0,
            'total_return': 0,
            'avg_return': 0,
            'max_water': 0
        }
    
    total_trades = len(trades)
    wins = len([t for t in trades if t['return'] > 0])
    
    total_invested = sum(t['num_buys'] * CAPITAL_PER_ENTRY for t in trades)
    total_profit = sum(t['num_buys'] * CAPITAL_PER_ENTRY * t['return'] / 100 for t in trades)
    total_return = (total_profit / total_invested * 100) if total_invested > 0 else 0
    
    avg_return = sum(t['return'] for t in trades) / total_trades
    max_water = max(t['num_buys'] for t in trades)
    
    return {
        'total_trades': total_trades,
        'win_rate': wins / total_trades * 100,
        'total_invested': total_invested,
        'total_profit': total_profit,
        'total_return': total_return,
        'avg_return': avg_return,
        'max_water': max_water
    }


def main():
    print("=" * 60)
    print("QQQ 전략 파라미터 최적화")
    print("=" * 60)
    
    # 데이터 로드
    print("\n📊 데이터 로드 중...")
    df = load_data('QQQ')
    print(f"✅ {len(df)}일 데이터 로드 완료")
    print(f"📅 기간: {df.index[0].date()} ~ {df.index[-1].date()}")
    
    # 파라미터 범위
    rsi_oversold_range = [30, 35, 40]          # 3
    rsi_buy_exit_range = [35, 40, 45]          # 3
    rsi_overbought_range = [65, 70, 75, 80]    # 4
    rsi_sell_exit_range = [40, 45, 50, 55]     # 4
    golden_cross_range = [True, False]          # 2
    ma_range = [(40, 200), (50, 200)]          # 2
    
    total_combinations = (len(rsi_oversold_range) * len(rsi_buy_exit_range) * 
                         len(rsi_overbought_range) * len(rsi_sell_exit_range) * 
                         len(golden_cross_range) * len(ma_range))
    
    print(f"\n🔍 테스트할 조합: {total_combinations}개")
    
    results = []
    
    for ma_short, ma_long in tqdm(ma_range, desc="MA 조합"):
        # MA 지표 추가
        df_ma = add_ma_indicators(df, ma_short, ma_long)
        
        for rsi_oversold in rsi_oversold_range:
            for rsi_buy_exit in rsi_buy_exit_range:
                # rsi_buy_exit는 rsi_oversold보다 커야 함
                if rsi_buy_exit <= rsi_oversold:
                    continue
                    
                for rsi_overbought in rsi_overbought_range:
                    for rsi_sell_exit in rsi_sell_exit_range:
                        # rsi_sell_exit는 rsi_overbought보다 작아야 함
                        if rsi_sell_exit >= rsi_overbought:
                            continue
                            
                        for use_gc in golden_cross_range:
                            # 시그널 찾기
                            buy_signals = find_buy_signals(df_ma, rsi_oversold, rsi_buy_exit, use_gc)
                            sell_signals = find_sell_signals(df_ma, rsi_overbought, rsi_sell_exit)
                            
                            # 시뮬레이션
                            trades, positions = simulate_trades(df_ma, buy_signals, sell_signals)
                            
                            # 성과 계산
                            perf = calculate_performance(trades)
                            
                            results.append({
                                'rsi_oversold': rsi_oversold,
                                'rsi_buy_exit': rsi_buy_exit,
                                'rsi_overbought': rsi_overbought,
                                'rsi_sell_exit': rsi_sell_exit,
                                'golden_cross': 'ON' if use_gc else 'OFF',
                                'ma_short': ma_short,
                                'ma_long': ma_long,
                                'trades': perf['total_trades'],
                                'win_rate': perf['win_rate'],
                                'invested': perf['total_invested'],
                                'profit': perf['total_profit'],
                                'return_pct': perf['total_return'],
                                'avg_return': perf['avg_return'],
                                'max_water': perf['max_water']
                            })
    
    # 결과 정리
    results_df = pd.DataFrame(results)
    
    # 필터: 최소 10회 이상 거래
    results_df = results_df[results_df['trades'] >= 10]
    
    # 정렬: 금액 수익률 기준
    results_df = results_df.sort_values('profit', ascending=False)
    
    # 저장
    results_df.to_csv('data/qqq_optimization_results.csv', index=False)
    
    print("\n" + "=" * 60)
    print("🏆 상위 10개 전략")
    print("=" * 60)
    
    for i, row in results_df.head(10).iterrows():
        print(f"\n#{results_df.index.get_loc(i) + 1}")
        print(f"  RSI: {row['rsi_oversold']}/{row['rsi_buy_exit']} → {row['rsi_overbought']}/{row['rsi_sell_exit']}")
        print(f"  MA: {row['ma_short']}/{row['ma_long']}, GC: {row['golden_cross']}")
        print(f"  거래: {row['trades']}회, 승률: {row['win_rate']:.0f}%")
        print(f"  투자금: ${row['invested']:,.0f}, 손익: ${row['profit']:+,.0f}")
        print(f"  금액 수익률: {row['return_pct']:+.1f}%, 최대 물타기: {row['max_water']}회")
    
    print("\n" + "=" * 60)
    print("📊 현재 전략 vs 최적 전략 비교")
    print("=" * 60)
    
    # 현재 전략 찾기
    current = results_df[
        (results_df['rsi_oversold'] == 35) &
        (results_df['rsi_buy_exit'] == 40) &
        (results_df['rsi_overbought'] == 70) &
        (results_df['rsi_sell_exit'] == 45) &
        (results_df['golden_cross'] == 'OFF') &
        (results_df['ma_short'] == 40)
    ]
    
    if not current.empty:
        curr = current.iloc[0]
        best = results_df.iloc[0]
        
        print(f"\n현재 전략 (RSI 35/40 → 70/45, MA40/200, GC OFF):")
        print(f"  거래: {curr['trades']}회, 승률: {curr['win_rate']:.0f}%")
        print(f"  손익: ${curr['profit']:+,.0f}, 수익률: {curr['return_pct']:+.1f}%")
        
        print(f"\n최적 전략 (RSI {best['rsi_oversold']}/{best['rsi_buy_exit']} → {best['rsi_overbought']}/{best['rsi_sell_exit']}, MA{best['ma_short']}/{best['ma_long']}, GC {best['golden_cross']}):")
        print(f"  거래: {best['trades']}회, 승률: {best['win_rate']:.0f}%")
        print(f"  손익: ${best['profit']:+,.0f}, 수익률: {best['return_pct']:+.1f}%")
        
        improvement = best['profit'] - curr['profit']
        print(f"\n🎯 개선 효과: ${improvement:+,.0f} ({(improvement/curr['profit']*100) if curr['profit'] != 0 else 0:+.1f}%)")
    
    print(f"\n✅ 결과 저장: data/qqq_optimization_results.csv")


if __name__ == "__main__":
    main()

