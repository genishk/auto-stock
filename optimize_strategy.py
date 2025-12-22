"""
Auto-Stock (QQQ) 전략 최적화
- RSI 파라미터 + 골든크로스 + 손절라인 최적화
- 10년 데이터 기반
- 약 800개 조합 테스트
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm

from src.data.cache import DataCache
from src.data.fetcher import DataFetcher
from src.data.validator import DataValidator
from src.features.technical import TechnicalIndicators
from src.discovery.validated_patterns import VALIDATED_PATTERNS
from src.utils.helpers import load_config


def load_data():
    """10년 QQQ 데이터 로드"""
    config = load_config()
    ticker = 'QQQ'
    
    cache = DataCache(cache_dir='data/cache', max_age_hours=24)
    df = cache.get(ticker)
    if df is None:
        fetcher = DataFetcher([ticker])
        data = fetcher.fetch('10y')
        df = data[ticker]
        df, _ = DataValidator.validate(df, ticker)
        cache.set(ticker, df)
    
    ti = TechnicalIndicators(config.get('indicators', {}))
    df = ti.calculate_all(df)
    
    # 골든크로스용 MA
    df['MA40'] = df['Close'].rolling(window=40).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['golden_cross'] = df['MA40'] > df['MA200']
    
    return df


def find_buy_signals(df, rsi_oversold, rsi_exit, use_gc=True):
    """매수 시그널 찾기"""
    buy_signals = []
    in_oversold = False
    last_date = None
    last_price = None
    
    for idx in range(len(df)):
        rsi = df['rsi'].iloc[idx]
        if pd.isna(rsi):
            continue
        
        gc_ok = True
        if use_gc:
            gc = df['golden_cross'].iloc[idx]
            gc_ok = gc if not pd.isna(gc) else False
        
        if rsi < rsi_oversold:
            in_oversold = True
            last_date = df.index[idx]
            last_price = df['Close'].iloc[idx]
        else:
            if in_oversold and rsi >= rsi_exit and last_date is not None and gc_ok:
                buy_signals.append({
                    'confirm_date': df.index[idx],
                    'confirm_price': df['Close'].iloc[idx]
                })
                in_oversold = False
                last_date = None
    
    return buy_signals


def find_sell_signals(df, rsi_overbought, rsi_exit):
    """매도 시그널 찾기"""
    sell_signals = []
    in_overbought = False
    last_date = None
    
    for idx in range(len(df)):
        rsi = df['rsi'].iloc[idx]
        if pd.isna(rsi):
            continue
        
        if rsi > rsi_overbought:
            in_overbought = True
            last_date = df.index[idx]
        else:
            if in_overbought and rsi <= rsi_exit and last_date is not None:
                sell_signals.append({
                    'confirm_date': df.index[idx],
                    'confirm_price': df['Close'].iloc[idx]
                })
                in_overbought = False
                last_date = None
    
    return sell_signals


def simulate_trades(df, buy_signals, sell_signals, stop_loss):
    """물타기 전략 시뮬레이션 (수익일 때만 익절)"""
    all_buy_dates = {bs['confirm_date']: bs for bs in buy_signals}
    all_sell_dates = {ss['confirm_date']: ss for ss in sell_signals}
    
    trades = []
    positions = []
    
    for idx in range(len(df)):
        current_date = df.index[idx]
        current_price = df['Close'].iloc[idx]
        
        if positions:
            total_cost = sum(p['price'] for p in positions)
            avg_price = total_cost / len(positions)
            current_return = (current_price / avg_price - 1) * 100
            
            exit_reason = None
            exit_price = current_price
            
            # 손절은 무조건
            if current_return <= stop_loss:
                exit_reason = "손절"
            # 수익일 때만 익절
            elif current_date in all_sell_dates:
                sell_price = all_sell_dates[current_date]['confirm_price']
                sell_return = (sell_price / avg_price - 1) * 100
                if sell_return > 0:
                    exit_reason = "익절"
                    exit_price = sell_price
            
            if exit_reason:
                final_return = (exit_price / avg_price - 1) * 100
                trades.append({
                    'num_buys': len(positions),
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


def evaluate_params(df, rsi_os, rsi_buy_exit, rsi_ob, rsi_sell_exit, stop_loss, use_gc):
    """파라미터 조합 평가"""
    buy_signals = find_buy_signals(df, rsi_os, rsi_buy_exit, use_gc)
    sell_signals = find_sell_signals(df, rsi_ob, rsi_sell_exit)
    trades, current_pos = simulate_trades(df, buy_signals, sell_signals, stop_loss)
    
    if not trades:
        return None
    
    total_return = sum(t['return'] for t in trades)
    avg_return = total_return / len(trades)
    win_rate = len([t for t in trades if t['return'] > 0]) / len(trades) * 100
    num_trades = len(trades)
    current_holding = len(current_pos)
    
    return {
        'total_return': total_return,
        'avg_return': avg_return,
        'win_rate': win_rate,
        'num_trades': num_trades,
        'current_holding': current_holding
    }


def main():
    print("=" * 60)
    print("🔍 Auto-Stock (QQQ) 전략 최적화")
    print("=" * 60)
    
    print("\n📊 데이터 로딩 중...")
    df = load_data()
    print(f"데이터 기간: {df.index[0].date()} ~ {df.index[-1].date()}")
    print(f"데이터 포인트: {len(df):,}개 (약 {len(df)/252:.1f}년)")
    
    # 파라미터 범위 설정 (약 800개 조합)
    rsi_oversold_range = [30, 35, 40]           # 3개
    rsi_buy_exit_range = [35, 40, 45, 50, 55]   # 5개
    rsi_overbought_range = [70, 75, 80, 85]     # 4개
    rsi_sell_exit_range = [45, 50, 55, 60]      # 4개
    stop_loss_range = [-20, -25, -30, -35]      # 4개
    gc_options = [True, False]                   # 2개
    
    combinations = list(product(
        rsi_oversold_range,
        rsi_buy_exit_range,
        rsi_overbought_range,
        rsi_sell_exit_range,
        stop_loss_range,
        gc_options
    ))
    
    # 유효한 조합만 필터링 (매수탈출 > 과매도, 매도탈출 < 과매수)
    valid_combinations = [
        c for c in combinations 
        if c[1] > c[0] and c[3] < c[2]
    ]
    
    print(f"\n🔄 총 {len(valid_combinations):,}개 조합 테스트 중...")
    
    results = []
    for params in tqdm(valid_combinations, desc="최적화"):
        rsi_os, rsi_buy_exit, rsi_ob, rsi_sell_exit, stop_loss, use_gc = params
        
        result = evaluate_params(df, rsi_os, rsi_buy_exit, rsi_ob, rsi_sell_exit, stop_loss, use_gc)
        
        if result:
            results.append({
                'rsi_oversold': rsi_os,
                'rsi_buy_exit': rsi_buy_exit,
                'rsi_overbought': rsi_ob,
                'rsi_sell_exit': rsi_sell_exit,
                'stop_loss': stop_loss,
                'golden_cross': use_gc,
                **result
            })
    
    # 결과 정렬
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 60)
    print("📈 총 수익률 TOP 10 (골든크로스 ON)")
    print("=" * 60)
    
    gc_results = results_df[results_df['golden_cross'] == True].sort_values('total_return', ascending=False)
    for i, row in gc_results.head(10).iterrows():
        print(f"\n{gc_results.head(10).index.get_loc(i)+1}위: RSI {int(row['rsi_oversold'])}/{int(row['rsi_buy_exit'])}/{int(row['rsi_overbought'])}/{int(row['rsi_sell_exit'])}, 손절 {int(row['stop_loss'])}%")
        print(f"   총수익: {row['total_return']:+.1f}% | 평균: {row['avg_return']:+.1f}% | 승률: {row['win_rate']:.0f}% | 거래: {int(row['num_trades'])}회 | 보유중: {int(row['current_holding'])}회")
    
    print("\n" + "=" * 60)
    print("📈 총 수익률 TOP 10 (골든크로스 OFF)")
    print("=" * 60)
    
    no_gc_results = results_df[results_df['golden_cross'] == False].sort_values('total_return', ascending=False)
    for i, row in no_gc_results.head(10).iterrows():
        print(f"\n{no_gc_results.head(10).index.get_loc(i)+1}위: RSI {int(row['rsi_oversold'])}/{int(row['rsi_buy_exit'])}/{int(row['rsi_overbought'])}/{int(row['rsi_sell_exit'])}, 손절 {int(row['stop_loss'])}%")
        print(f"   총수익: {row['total_return']:+.1f}% | 평균: {row['avg_return']:+.1f}% | 승률: {row['win_rate']:.0f}% | 거래: {int(row['num_trades'])}회 | 보유중: {int(row['current_holding'])}회")
    
    print("\n" + "=" * 60)
    print("🏆 현재 보유 0회 중 최고 수익률")
    print("=" * 60)
    
    no_holding = results_df[results_df['current_holding'] == 0].sort_values('total_return', ascending=False)
    if len(no_holding) > 0:
        for i, row in no_holding.head(5).iterrows():
            gc_str = "✅ GC" if row['golden_cross'] else "❌ GC"
            print(f"\n{no_holding.head(5).index.get_loc(i)+1}위: RSI {int(row['rsi_oversold'])}/{int(row['rsi_buy_exit'])}/{int(row['rsi_overbought'])}/{int(row['rsi_sell_exit'])}, 손절 {int(row['stop_loss'])}% {gc_str}")
            print(f"   총수익: {row['total_return']:+.1f}% | 평균: {row['avg_return']:+.1f}% | 승률: {row['win_rate']:.0f}% | 거래: {int(row['num_trades'])}회")
    else:
        print("모든 조합에서 현재 포지션 보유 중")
    
    print("\n" + "=" * 60)
    print("📊 골든크로스 효과 비교")
    print("=" * 60)
    
    gc_avg = gc_results['total_return'].mean()
    no_gc_avg = no_gc_results['total_return'].mean()
    print(f"골든크로스 ON 평균 수익률: {gc_avg:+.1f}%")
    print(f"골든크로스 OFF 평균 수익률: {no_gc_avg:+.1f}%")
    print(f"차이: {gc_avg - no_gc_avg:+.1f}%p")
    
    # 최적 파라미터 추천
    print("\n" + "=" * 60)
    print("⭐ 추천 파라미터")
    print("=" * 60)
    
    # 골든크로스 + 보유 0회 + 높은 수익률
    best = gc_results[gc_results['current_holding'] == 0].head(1)
    if len(best) == 0:
        best = gc_results.head(1)
    
    if len(best) > 0:
        row = best.iloc[0]
        print(f"\n과매도 기준: RSI < {int(row['rsi_oversold'])}")
        print(f"매수 탈출: RSI ≥ {int(row['rsi_buy_exit'])}")
        print(f"과매수 기준: RSI > {int(row['rsi_overbought'])}")
        print(f"매도 탈출: RSI ≤ {int(row['rsi_sell_exit'])}")
        print(f"손절: {int(row['stop_loss'])}%")
        print(f"골든크로스: {'사용' if row['golden_cross'] else '미사용'}")
        print(f"\n예상 성과:")
        print(f"  - 총 수익률: {row['total_return']:+.1f}%")
        print(f"  - 평균 수익률: {row['avg_return']:+.1f}%")
        print(f"  - 승률: {row['win_rate']:.0f}%")
        print(f"  - 거래 횟수: {int(row['num_trades'])}회")


if __name__ == '__main__':
    main()

