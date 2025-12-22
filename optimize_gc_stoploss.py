"""
Auto-Stock (QQQ) 골든크로스 + 손절 최적화
- RSI: 35/40/70/45 고정
- 골든크로스 MA 조합 다양화
- 손절 없음 vs 있음 비교
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
    
    return df


def add_golden_cross(df, short_ma, long_ma):
    """골든크로스 계산"""
    df = df.copy()
    df[f'MA{short_ma}'] = df['Close'].rolling(window=short_ma).mean()
    df[f'MA{long_ma}'] = df['Close'].rolling(window=long_ma).mean()
    df['golden_cross'] = df[f'MA{short_ma}'] > df[f'MA{long_ma}']
    return df


def find_buy_signals(df, use_gc=True):
    """매수 시그널 찾기 (RSI 35/40 고정)"""
    rsi_oversold = 35
    rsi_exit = 40
    
    buy_signals = []
    in_oversold = False
    last_date = None
    
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
        else:
            if in_oversold and rsi >= rsi_exit and last_date is not None and gc_ok:
                buy_signals.append({
                    'confirm_date': df.index[idx],
                    'confirm_price': df['Close'].iloc[idx]
                })
                in_oversold = False
                last_date = None
    
    return buy_signals


def find_sell_signals(df):
    """매도 시그널 찾기 (RSI 70/45 고정)"""
    rsi_overbought = 70
    rsi_exit = 45
    
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


def simulate_trades(df, buy_signals, sell_signals, stop_loss=None):
    """
    물타기 전략 시뮬레이션 (수익일 때만 익절)
    stop_loss=None이면 손절 없음
    """
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
            
            # 손절 (있을 경우만)
            if stop_loss is not None and current_return <= stop_loss:
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


def evaluate(df, short_ma, long_ma, use_gc, stop_loss):
    """파라미터 조합 평가"""
    if use_gc:
        df = add_golden_cross(df, short_ma, long_ma)
    
    buy_signals = find_buy_signals(df, use_gc)
    sell_signals = find_sell_signals(df)
    trades, current_pos = simulate_trades(df, buy_signals, sell_signals, stop_loss)
    
    if not trades:
        return None
    
    total_return = sum(t['return'] for t in trades)
    avg_return = total_return / len(trades)
    win_rate = len([t for t in trades if t['return'] > 0]) / len(trades) * 100
    num_trades = len(trades)
    current_holding = len(current_pos)
    stoploss_count = len([t for t in trades if t['exit_reason'] == '손절'])
    
    return {
        'total_return': total_return,
        'avg_return': avg_return,
        'win_rate': win_rate,
        'num_trades': num_trades,
        'current_holding': current_holding,
        'stoploss_count': stoploss_count
    }


def main():
    print("=" * 60)
    print("🔍 Auto-Stock (QQQ) 골든크로스 + 손절 최적화")
    print("   RSI 고정: 35/40/70/45")
    print("=" * 60)
    
    print("\n📊 데이터 로딩 중...")
    df = load_data()
    print(f"데이터 기간: {df.index[0].date()} ~ {df.index[-1].date()}")
    print(f"데이터 포인트: {len(df):,}개 (약 {len(df)/252:.1f}년)")
    
    # 골든크로스 MA 조합
    short_ma_range = [20, 30, 40, 50, 60, 70]
    long_ma_range = [100, 150, 200]
    
    # 손절 옵션 (None = 손절 없음)
    stop_loss_options = [None, -20, -25, -30, -35]
    
    results = []
    
    # 1. 골든크로스 OFF 테스트
    print("\n🔄 골든크로스 OFF 테스트...")
    for stop_loss in tqdm(stop_loss_options, desc="손절"):
        result = evaluate(df.copy(), 0, 0, False, stop_loss)
        if result:
            results.append({
                'short_ma': 0,
                'long_ma': 0,
                'use_gc': False,
                'stop_loss': stop_loss,
                **result
            })
    
    # 2. 골든크로스 ON 테스트
    print("\n🔄 골든크로스 ON 테스트...")
    gc_combinations = [(s, l) for s in short_ma_range for l in long_ma_range if s < l]
    
    for short_ma, long_ma in tqdm(gc_combinations, desc="MA 조합"):
        for stop_loss in stop_loss_options:
            result = evaluate(df.copy(), short_ma, long_ma, True, stop_loss)
            if result:
                results.append({
                    'short_ma': short_ma,
                    'long_ma': long_ma,
                    'use_gc': True,
                    'stop_loss': stop_loss,
                    **result
                })
    
    results_df = pd.DataFrame(results)
    
    # ===== 결과 출력 =====
    print("\n" + "=" * 60)
    print("📈 골든크로스 OFF 결과")
    print("=" * 60)
    
    gc_off = results_df[results_df['use_gc'] == False].sort_values('total_return', ascending=False)
    for _, row in gc_off.iterrows():
        sl_str = f"{int(row['stop_loss'])}%" if pd.notna(row['stop_loss']) else "없음"
        print(f"손절 {sl_str:>5}: 총수익 {row['total_return']:+.1f}% | 평균 {row['avg_return']:+.1f}% | 승률 {row['win_rate']:.0f}% | 거래 {int(row['num_trades'])}회 | 손절 {int(row['stoploss_count'])}회 | 보유 {int(row['current_holding'])}회")
    
    print("\n" + "=" * 60)
    print("📈 골든크로스 ON - MA 조합별 TOP 10 (손절 없음)")
    print("=" * 60)
    
    gc_on_no_sl = results_df[(results_df['use_gc'] == True) & (results_df['stop_loss'].isna())].sort_values('total_return', ascending=False)
    for i, (_, row) in enumerate(gc_on_no_sl.head(10).iterrows()):
        print(f"{i+1:2}위: MA{int(row['short_ma'])}/{int(row['long_ma'])} | 총수익 {row['total_return']:+.1f}% | 평균 {row['avg_return']:+.1f}% | 승률 {row['win_rate']:.0f}% | 거래 {int(row['num_trades'])}회 | 보유 {int(row['current_holding'])}회")
    
    print("\n" + "=" * 60)
    print("📈 골든크로스 ON - MA 조합별 TOP 10 (손절 -25%)")
    print("=" * 60)
    
    gc_on_sl25 = results_df[(results_df['use_gc'] == True) & (results_df['stop_loss'] == -25)].sort_values('total_return', ascending=False)
    for i, (_, row) in enumerate(gc_on_sl25.head(10).iterrows()):
        print(f"{i+1:2}위: MA{int(row['short_ma'])}/{int(row['long_ma'])} | 총수익 {row['total_return']:+.1f}% | 평균 {row['avg_return']:+.1f}% | 승률 {row['win_rate']:.0f}% | 거래 {int(row['num_trades'])}회 | 손절 {int(row['stoploss_count'])}회 | 보유 {int(row['current_holding'])}회")
    
    print("\n" + "=" * 60)
    print("🏆 현재 보유 0회 중 최고 수익률")
    print("=" * 60)
    
    no_holding = results_df[results_df['current_holding'] == 0].sort_values('total_return', ascending=False)
    for i, (_, row) in enumerate(no_holding.head(10).iterrows()):
        gc_str = f"MA{int(row['short_ma'])}/{int(row['long_ma'])}" if row['use_gc'] else "OFF"
        sl_str = f"{int(row['stop_loss'])}%" if pd.notna(row['stop_loss']) else "없음"
        print(f"{i+1:2}위: GC {gc_str:>10} | 손절 {sl_str:>5} | 총수익 {row['total_return']:+.1f}% | 평균 {row['avg_return']:+.1f}% | 승률 {row['win_rate']:.0f}% | 거래 {int(row['num_trades'])}회")
    
    print("\n" + "=" * 60)
    print("📊 손절 있음 vs 없음 비교 (골든크로스 OFF)")
    print("=" * 60)
    
    gc_off_no_sl = gc_off[gc_off['stop_loss'].isna()].iloc[0] if len(gc_off[gc_off['stop_loss'].isna()]) > 0 else None
    gc_off_sl25 = gc_off[gc_off['stop_loss'] == -25].iloc[0] if len(gc_off[gc_off['stop_loss'] == -25]) > 0 else None
    
    if gc_off_no_sl is not None:
        print(f"손절 없음: 총수익 {gc_off_no_sl['total_return']:+.1f}% | 승률 {gc_off_no_sl['win_rate']:.0f}% | 보유 {int(gc_off_no_sl['current_holding'])}회")
    if gc_off_sl25 is not None:
        print(f"손절 -25%: 총수익 {gc_off_sl25['total_return']:+.1f}% | 승률 {gc_off_sl25['win_rate']:.0f}% | 보유 {int(gc_off_sl25['current_holding'])}회")
    
    print("\n" + "=" * 60)
    print("⭐ 최종 추천")
    print("=" * 60)
    
    # 보유 0회 + 높은 수익률 기준
    best = no_holding.iloc[0] if len(no_holding) > 0 else results_df.sort_values('total_return', ascending=False).iloc[0]
    
    gc_str = f"MA{int(best['short_ma'])}/{int(best['long_ma'])}" if best['use_gc'] else "미사용"
    sl_str = f"{int(best['stop_loss'])}%" if pd.notna(best['stop_loss']) else "없음"
    
    print(f"\nRSI: 35/40/70/45 (고정)")
    print(f"골든크로스: {gc_str}")
    print(f"손절: {sl_str}")
    print(f"\n예상 성과:")
    print(f"  - 총 수익률: {best['total_return']:+.1f}%")
    print(f"  - 평균 수익률: {best['avg_return']:+.1f}%")
    print(f"  - 승률: {best['win_rate']:.0f}%")
    print(f"  - 거래 횟수: {int(best['num_trades'])}회")
    print(f"  - 현재 보유: {int(best['current_holding'])}회")


if __name__ == '__main__':
    main()

