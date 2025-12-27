"""
AAPL, SMH 전략 파라미터 최적화
- 동일 금액($1,000) 기준 계산
- 10년 데이터 사용
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


def load_data(ticker):
    """데이터 로드"""
    config = load_config()
    cache = DataCache(
        cache_dir=str(Path(__file__).parent / config['data']['cache']['directory']),
        max_age_hours=24
    )
    
    df = cache.get(ticker)
    if df is None:
        fetcher = DataFetcher([ticker])
        data = fetcher.fetch('10y')
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
        else:
            if in_oversold and rsi >= rsi_exit and last_signal_date is not None:
                if golden_cross_ok:
                    buy_signals.append({
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
    
    for idx in range(len(df)):
        rsi = df['rsi'].iloc[idx]
        if pd.isna(rsi):
            continue
        
        if rsi > rsi_overbought:
            in_overbought = True
            last_signal_date = df.index[idx]
        else:
            if in_overbought and rsi <= rsi_exit and last_signal_date is not None:
                sell_signals.append({
                    'confirm_date': df.index[idx],
                    'confirm_price': df['Close'].iloc[idx]
                })
                in_overbought = False
                last_signal_date = None
    
    return sell_signals


def simulate_trades(df, buy_signals, sell_signals):
    """거래 시뮬레이션 (동일 금액 기준, profit_only)"""
    all_buy_dates = {bs['confirm_date']: bs for bs in buy_signals}
    all_sell_dates = {ss['confirm_date']: ss for ss in sell_signals}
    
    trades = []
    positions = []
    
    for idx in range(len(df)):
        current_date = df.index[idx]
        current_price = df['Close'].iloc[idx]
        
        if positions:
            total_invested = len(positions) * CAPITAL_PER_ENTRY
            total_quantity = sum(CAPITAL_PER_ENTRY / p['price'] for p in positions)
            avg_price = total_invested / total_quantity
            
            if current_date in all_sell_dates:
                sell_price = all_sell_dates[current_date]['confirm_price']
                sell_return = (sell_price / avg_price - 1) * 100
                if sell_return > 0:
                    final_return = (sell_price / avg_price - 1) * 100
                    trades.append({
                        'num_buys': len(positions),
                        'return': final_return
                    })
                    positions = []
        
        if current_date in all_buy_dates:
            positions.append({
                'date': current_date,
                'price': all_buy_dates[current_date]['confirm_price']
            })
    
    return trades, positions


def calculate_performance(trades):
    """성과 계산"""
    if not trades:
        return None
    
    total_trades = len(trades)
    wins = len([t for t in trades if t['return'] > 0])
    
    total_invested = sum(t['num_buys'] * CAPITAL_PER_ENTRY for t in trades)
    total_profit = sum(t['num_buys'] * CAPITAL_PER_ENTRY * t['return'] / 100 for t in trades)
    total_return = (total_profit / total_invested * 100) if total_invested > 0 else 0
    
    max_water = max(t['num_buys'] for t in trades)
    
    return {
        'total_trades': total_trades,
        'win_rate': wins / total_trades * 100,
        'total_invested': total_invested,
        'total_profit': total_profit,
        'total_return': total_return,
        'max_water': max_water
    }


def optimize_ticker(ticker):
    """종목 최적화"""
    print(f"\n{'='*60}")
    print(f"🔍 {ticker} 전략 파라미터 최적화")
    print(f"{'='*60}")
    
    # 데이터 로드
    print(f"\n📊 데이터 로드 중...")
    df = load_data(ticker)
    print(f"✅ {len(df)}일 데이터 로드 완료")
    print(f"📅 기간: {df.index[0].date()} ~ {df.index[-1].date()}")
    
    # 파라미터 범위
    rsi_oversold_range = [30, 35, 40]
    rsi_buy_exit_range = [35, 40, 45]
    rsi_overbought_range = [65, 70, 75, 80]
    rsi_sell_exit_range = [40, 45, 50, 55]
    golden_cross_range = [True, False]
    ma_range = [(40, 200), (50, 200)]
    
    results = []
    
    for ma_short, ma_long in tqdm(ma_range, desc=f"{ticker} MA"):
        df_ma = add_ma_indicators(df, ma_short, ma_long)
        
        for rsi_oversold in rsi_oversold_range:
            for rsi_buy_exit in rsi_buy_exit_range:
                if rsi_buy_exit <= rsi_oversold:
                    continue
                    
                for rsi_overbought in rsi_overbought_range:
                    for rsi_sell_exit in rsi_sell_exit_range:
                        if rsi_sell_exit >= rsi_overbought:
                            continue
                            
                        for use_gc in golden_cross_range:
                            buy_signals = find_buy_signals(df_ma, rsi_oversold, rsi_buy_exit, use_gc)
                            sell_signals = find_sell_signals(df_ma, rsi_overbought, rsi_sell_exit)
                            trades, positions = simulate_trades(df_ma, buy_signals, sell_signals)
                            perf = calculate_performance(trades)
                            
                            if perf and perf['total_trades'] >= 5:
                                results.append({
                                    'rsi_oversold': rsi_oversold,
                                    'rsi_buy_exit': rsi_buy_exit,
                                    'rsi_overbought': rsi_overbought,
                                    'rsi_sell_exit': rsi_sell_exit,
                                    'golden_cross': 'ON' if use_gc else 'OFF',
                                    'ma_short': ma_short,
                                    **perf
                                })
    
    # 결과 정리
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('total_profit', ascending=False)
    
    # 저장
    results_df.to_csv(f'data/{ticker.lower()}_optimization_results.csv', index=False)
    
    # 상위 5개 출력
    print(f"\n🏆 {ticker} 상위 5개 전략")
    print("-" * 60)
    
    for i, row in results_df.head(5).iterrows():
        print(f"\n#{results_df.index.get_loc(i) + 1}")
        print(f"  RSI: {row['rsi_oversold']}/{row['rsi_buy_exit']} → {row['rsi_overbought']}/{row['rsi_sell_exit']}")
        print(f"  MA: {row['ma_short']}/200, GC: {row['golden_cross']}")
        print(f"  거래: {row['total_trades']}회, 승률: {row['win_rate']:.0f}%")
        print(f"  손익: ${row['total_profit']:+,.0f}, 수익률: {row['total_return']:+.1f}%")
        print(f"  최대 물타기: {row['max_water']}회")
    
    # 현재 QQQ 전략과 비교
    print(f"\n📊 현재 QQQ 전략 (35/40 → 75/50) vs 최적 전략")
    print("-" * 60)
    
    current = results_df[
        (results_df['rsi_oversold'] == 35) &
        (results_df['rsi_buy_exit'] == 40) &
        (results_df['rsi_overbought'] == 75) &
        (results_df['rsi_sell_exit'] == 50) &
        (results_df['golden_cross'] == 'OFF')
    ]
    
    if not current.empty:
        curr = current.iloc[0]
        best = results_df.iloc[0]
        
        print(f"\n현재 전략 (QQQ 기준):")
        print(f"  거래: {curr['total_trades']}회, 손익: ${curr['total_profit']:+,.0f} ({curr['total_return']:+.1f}%)")
        
        print(f"\n최적 전략 ({best['rsi_oversold']}/{best['rsi_buy_exit']} → {best['rsi_overbought']}/{best['rsi_sell_exit']}):")
        print(f"  거래: {best['total_trades']}회, 손익: ${best['total_profit']:+,.0f} ({best['total_return']:+.1f}%)")
        
        improvement = best['total_profit'] - curr['total_profit']
        print(f"\n🎯 개선 효과: ${improvement:+,.0f}")
    else:
        best = results_df.iloc[0]
        print(f"\n최적 전략: RSI {best['rsi_oversold']}/{best['rsi_buy_exit']} → {best['rsi_overbought']}/{best['rsi_sell_exit']}")
        print(f"  거래: {best['total_trades']}회, 손익: ${best['total_profit']:+,.0f} ({best['total_return']:+.1f}%)")
    
    return results_df


def main():
    print("=" * 60)
    print("🚀 AAPL & SMH 전략 파라미터 최적화")
    print("=" * 60)
    print("기준: 동일 금액($1,000), profit_only")
    
    # AAPL 최적화
    aapl_results = optimize_ticker('AAPL')
    
    # SMH 최적화
    smh_results = optimize_ticker('SMH')
    
    # 최종 요약
    print("\n" + "=" * 60)
    print("📋 최종 요약")
    print("=" * 60)
    
    print("\n🍎 AAPL 최적 전략:")
    best_aapl = aapl_results.iloc[0]
    print(f"   RSI: {best_aapl['rsi_oversold']}/{best_aapl['rsi_buy_exit']} → {best_aapl['rsi_overbought']}/{best_aapl['rsi_sell_exit']}")
    print(f"   GC: {best_aapl['golden_cross']}")
    print(f"   손익: ${best_aapl['total_profit']:+,.0f} ({best_aapl['total_return']:+.1f}%)")
    
    print("\n💎 SMH 최적 전략:")
    best_smh = smh_results.iloc[0]
    print(f"   RSI: {best_smh['rsi_oversold']}/{best_smh['rsi_buy_exit']} → {best_smh['rsi_overbought']}/{best_smh['rsi_sell_exit']}")
    print(f"   GC: {best_smh['golden_cross']}")
    print(f"   손익: ${best_smh['total_profit']:+,.0f} ({best_smh['total_return']:+.1f}%)")


if __name__ == "__main__":
    main()

