"""
WMT, GLD 최적화 - 거래수 늘리는 버전
다양한 기준으로 TOP 조합 비교
"""

import pandas as pd
import numpy as np
import yfinance as yf
from itertools import product

# ===== Wilder's Smoothing RSI (대시보드와 동일) =====
def calculate_rsi_wilder(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    for i in range(period, len(prices)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period-1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period-1) + loss.iloc[i]) / period
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def backtest_strategy(df, rsi_oversold, rsi_buy_exit, rsi_overbought, rsi_sell_exit, capital=1000):
    df = df.copy()
    df['rsi'] = calculate_rsi_wilder(df['Close'], 14)
    
    buy_signals = []
    in_oversold = False
    
    for idx in range(len(df)):
        rsi = df['rsi'].iloc[idx]
        if pd.isna(rsi):
            continue
        if rsi < rsi_oversold:
            in_oversold = True
        else:
            if in_oversold and rsi >= rsi_buy_exit:
                buy_signals.append({'date': df.index[idx], 'price': df['Close'].iloc[idx]})
                in_oversold = False
    
    sell_signals = []
    in_overbought = False
    
    for idx in range(len(df)):
        rsi = df['rsi'].iloc[idx]
        if pd.isna(rsi):
            continue
        if rsi > rsi_overbought:
            in_overbought = True
        else:
            if in_overbought and rsi <= rsi_sell_exit:
                sell_signals.append({'date': df.index[idx], 'price': df['Close'].iloc[idx]})
                in_overbought = False
    
    all_buy_dates = {bs['date']: bs for bs in buy_signals}
    all_sell_dates = {ss['date']: ss for ss in sell_signals}
    
    trades = []
    positions = []
    
    for idx in range(len(df)):
        current_date = df.index[idx]
        
        if positions:
            n = len(positions)
            total_inv = n * capital
            total_qty = sum(capital / p['price'] for p in positions)
            avg_price = total_inv / total_qty
            
            if current_date in all_sell_dates:
                sell_price = all_sell_dates[current_date]['price']
                sell_return = (sell_price / avg_price - 1) * 100
                if sell_return > 0:
                    trades.append({
                        'num_buys': n,
                        'return': sell_return,
                        'profit': total_inv * sell_return / 100
                    })
                    positions = []
        
        if current_date in all_buy_dates:
            positions.append({'date': current_date, 'price': all_buy_dates[current_date]['price']})
    
    if not trades:
        return None
    
    total_invested = sum(t['num_buys'] * capital for t in trades)
    total_profit = sum(t['profit'] for t in trades)
    
    return {
        'trades': len(trades),
        'total_return': (total_profit / total_invested * 100) if total_invested > 0 else 0,
        'max_water': max(t['num_buys'] for t in trades),
        'avg_water': np.mean([t['num_buys'] for t in trades]),
        'win_rate': 100.0
    }


def optimize_ticker(ticker, df):
    print(f"\n{'='*80}")
    print(f"🔍 {ticker} 상세 최적화 분석")
    print(f"{'='*80}")
    print(f"데이터: {df.index[0].date()} ~ {df.index[-1].date()} ({len(df)}일, 약 {len(df)/252:.1f}년)")
    
    oversold_range = [25, 30, 35, 40, 45]
    buy_exit_range = [35, 40, 45, 50, 55, 60]
    overbought_range = [55, 60, 65, 70, 75, 80]
    sell_exit_range = [40, 45, 50, 55, 60]
    
    results = []
    
    for oversold, buy_exit, overbought, sell_exit in product(
        oversold_range, buy_exit_range, overbought_range, sell_exit_range
    ):
        if buy_exit <= oversold or sell_exit >= overbought:
            continue
        
        result = backtest_strategy(df, oversold, buy_exit, overbought, sell_exit)
        
        if result and result['trades'] >= 3:
            results.append({
                'oversold': oversold,
                'buy_exit': buy_exit,
                'overbought': overbought,
                'sell_exit': sell_exit,
                **result
            })
    
    if not results:
        print(f"❌ 조건 만족하는 조합 없음")
        return
    
    # ===== 1. 수익률 기준 TOP 10 =====
    by_return = sorted(results, key=lambda x: x['total_return'], reverse=True)
    print(f"\n📊 수익률 기준 TOP 10:")
    print("-" * 90)
    print(f"{'순위':^4} {'매수(과매도/탈출)':^16} {'매도(과매수/탈출)':^16} {'거래수':^8} {'수익률':^10} {'최대물타기':^10} {'연간거래':^8}")
    print("-" * 90)
    for i, r in enumerate(by_return[:10], 1):
        annual = r['trades'] / 10
        print(f"{i:^4} {r['oversold']:>6}/{r['buy_exit']:<6} {r['overbought']:>8}/{r['sell_exit']:<6} "
              f"{r['trades']:^8} {r['total_return']:>+8.1f}% {r['max_water']:^10} {annual:^8.1f}")
    
    # ===== 2. 거래수 기준 TOP 10 =====
    by_trades = sorted(results, key=lambda x: (x['trades'], x['total_return']), reverse=True)
    print(f"\n📊 거래수 기준 TOP 10:")
    print("-" * 90)
    print(f"{'순위':^4} {'매수(과매도/탈출)':^16} {'매도(과매수/탈출)':^16} {'거래수':^8} {'수익률':^10} {'최대물타기':^10} {'연간거래':^8}")
    print("-" * 90)
    for i, r in enumerate(by_trades[:10], 1):
        annual = r['trades'] / 10
        print(f"{i:^4} {r['oversold']:>6}/{r['buy_exit']:<6} {r['overbought']:>8}/{r['sell_exit']:<6} "
              f"{r['trades']:^8} {r['total_return']:>+8.1f}% {r['max_water']:^10} {annual:^8.1f}")
    
    # ===== 3. 물타기 최소 기준 TOP 10 =====
    by_water = sorted(results, key=lambda x: (x['max_water'], -x['total_return']))
    print(f"\n📊 물타기 최소 기준 TOP 10:")
    print("-" * 90)
    print(f"{'순위':^4} {'매수(과매도/탈출)':^16} {'매도(과매수/탈출)':^16} {'거래수':^8} {'수익률':^10} {'최대물타기':^10} {'연간거래':^8}")
    print("-" * 90)
    for i, r in enumerate(by_water[:10], 1):
        annual = r['trades'] / 10
        print(f"{i:^4} {r['oversold']:>6}/{r['buy_exit']:<6} {r['overbought']:>8}/{r['sell_exit']:<6} "
              f"{r['trades']:^8} {r['total_return']:>+8.1f}% {r['max_water']:^10} {annual:^8.1f}")
    
    # ===== 4. 균형 기준 (거래수 * 수익률 / 물타기) =====
    by_balanced = sorted(results, key=lambda x: x['trades'] * x['total_return'] / max(x['max_water'], 1), reverse=True)
    print(f"\n📊 균형 기준 TOP 10 (거래수 × 수익률 / 물타기):")
    print("-" * 90)
    print(f"{'순위':^4} {'매수(과매도/탈출)':^16} {'매도(과매수/탈출)':^16} {'거래수':^8} {'수익률':^10} {'최대물타기':^10} {'연간거래':^8}")
    print("-" * 90)
    for i, r in enumerate(by_balanced[:10], 1):
        annual = r['trades'] / 10
        print(f"{i:^4} {r['oversold']:>6}/{r['buy_exit']:<6} {r['overbought']:>8}/{r['sell_exit']:<6} "
              f"{r['trades']:^8} {r['total_return']:>+8.1f}% {r['max_water']:^10} {annual:^8.1f}")
    
    # ===== 5. 거래 늘리면서 물타기 5회 이하 제한 =====
    filtered = [r for r in results if r['max_water'] <= 5]
    if filtered:
        by_trades_filtered = sorted(filtered, key=lambda x: (x['trades'], x['total_return']), reverse=True)
        print(f"\n📊 거래 최대화 (물타기 ≤5회 제한) TOP 10:")
        print("-" * 90)
        print(f"{'순위':^4} {'매수(과매도/탈출)':^16} {'매도(과매수/탈출)':^16} {'거래수':^8} {'수익률':^10} {'최대물타기':^10} {'연간거래':^8}")
        print("-" * 90)
        for i, r in enumerate(by_trades_filtered[:10], 1):
            annual = r['trades'] / 10
            print(f"{i:^4} {r['oversold']:>6}/{r['buy_exit']:<6} {r['overbought']:>8}/{r['sell_exit']:<6} "
                  f"{r['trades']:^8} {r['total_return']:>+8.1f}% {r['max_water']:^10} {annual:^8.1f}")
    
    # ===== 추천 전략 =====
    print(f"\n{'='*80}")
    print(f"💡 {ticker} 추천 전략")
    print(f"{'='*80}")
    
    # 수익률 최고
    best_return = by_return[0]
    print(f"\n🏆 수익률 최고:")
    print(f"   RSI < {best_return['oversold']} → ≥ {best_return['buy_exit']} (매수)")
    print(f"   RSI > {best_return['overbought']} → ≤ {best_return['sell_exit']} (매도)")
    print(f"   → 거래 {best_return['trades']}회, 수익률 {best_return['total_return']:+.1f}%, 물타기 최대 {best_return['max_water']}회")
    
    # 거래수 최고
    best_trades = by_trades[0]
    print(f"\n📈 거래수 최고:")
    print(f"   RSI < {best_trades['oversold']} → ≥ {best_trades['buy_exit']} (매수)")
    print(f"   RSI > {best_trades['overbought']} → ≤ {best_trades['sell_exit']} (매도)")
    print(f"   → 거래 {best_trades['trades']}회, 수익률 {best_trades['total_return']:+.1f}%, 물타기 최대 {best_trades['max_water']}회")
    
    # 균형
    best_balanced = by_balanced[0]
    print(f"\n⚖️ 균형 (추천):")
    print(f"   RSI < {best_balanced['oversold']} → ≥ {best_balanced['buy_exit']} (매수)")
    print(f"   RSI > {best_balanced['overbought']} → ≤ {best_balanced['sell_exit']} (매도)")
    print(f"   → 거래 {best_balanced['trades']}회, 수익률 {best_balanced['total_return']:+.1f}%, 물타기 최대 {best_balanced['max_water']}회")
    
    if filtered:
        best_safe = by_trades_filtered[0]
        print(f"\n🛡️ 안전 (물타기 제한):")
        print(f"   RSI < {best_safe['oversold']} → ≥ {best_safe['buy_exit']} (매수)")
        print(f"   RSI > {best_safe['overbought']} → ≤ {best_safe['sell_exit']} (매도)")
        print(f"   → 거래 {best_safe['trades']}회, 수익률 {best_safe['total_return']:+.1f}%, 물타기 최대 {best_safe['max_water']}회")


def main():
    print("=" * 80)
    print("🏪 WMT, GLD 상세 최적화 분석 (거래수 늘리기)")
    print("=" * 80)
    
    tickers = ['WMT', 'GLD']
    
    for ticker in tickers:
        print(f"\n📥 {ticker} 데이터 로드 중...")
        df = yf.download(ticker, period='10y', progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if len(df) > 0:
            optimize_ticker(ticker, df)


if __name__ == '__main__':
    main()

