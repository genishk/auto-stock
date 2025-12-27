"""
WMT, GLD 최적화 스크립트
대시보드와 동일한 Wilder's Smoothing RSI 사용
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from itertools import product

# ===== Wilder's Smoothing RSI (대시보드와 동일) =====
def calculate_rsi_wilder(prices: pd.Series, period: int = 14) -> pd.Series:
    """대시보드와 동일한 Wilder's Smoothing RSI 계산"""
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


def backtest_strategy(df, rsi_oversold, rsi_buy_exit, rsi_overbought, rsi_sell_exit, capital=1000):
    """백테스트 실행"""
    df = df.copy()
    df['rsi'] = calculate_rsi_wilder(df['Close'], 14)
    
    # 매수 시그널 찾기
    buy_signals = []
    in_oversold = False
    last_date = None
    last_price = None
    
    for idx in range(len(df)):
        rsi = df['rsi'].iloc[idx]
        if pd.isna(rsi):
            continue
        
        if rsi < rsi_oversold:
            in_oversold = True
            last_date = df.index[idx]
            last_price = df['Close'].iloc[idx]
        else:
            if in_oversold and rsi >= rsi_buy_exit and last_date is not None:
                buy_signals.append({
                    'date': df.index[idx],
                    'price': df['Close'].iloc[idx]
                })
                in_oversold = False
                last_date = None
    
    # 매도 시그널 찾기
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
                sell_signals.append({
                    'date': df.index[idx],
                    'price': df['Close'].iloc[idx]
                })
                in_overbought = False
    
    # 거래 시뮬레이션
    all_buy_dates = {bs['date']: bs for bs in buy_signals}
    all_sell_dates = {ss['date']: ss for ss in sell_signals}
    
    trades = []
    positions = []
    
    for idx in range(len(df)):
        current_date = df.index[idx]
        current_price = df['Close'].iloc[idx]
        
        if positions:
            n = len(positions)
            total_inv = n * capital
            total_qty = sum(capital / p['price'] for p in positions)
            avg_price = total_inv / total_qty
            
            if current_date in all_sell_dates:
                sell_price = all_sell_dates[current_date]['price']
                sell_return = (sell_price / avg_price - 1) * 100
                if sell_return > 0:  # profit_only
                    trades.append({
                        'num_buys': n,
                        'avg_price': avg_price,
                        'exit_price': sell_price,
                        'return': sell_return,
                        'profit': total_inv * sell_return / 100
                    })
                    positions = []
        
        if current_date in all_buy_dates:
            positions.append({
                'date': current_date,
                'price': all_buy_dates[current_date]['price']
            })
    
    # 결과 계산
    if not trades:
        return {
            'trades': 0,
            'total_return': 0,
            'total_profit': 0,
            'max_water': 0,
            'avg_water': 0,
            'win_rate': 0
        }
    
    total_invested = sum(t['num_buys'] * capital for t in trades)
    total_profit = sum(t['profit'] for t in trades)
    total_return = (total_profit / total_invested * 100) if total_invested > 0 else 0
    
    return {
        'trades': len(trades),
        'total_return': total_return,
        'total_profit': total_profit,
        'max_water': max(t['num_buys'] for t in trades),
        'avg_water': np.mean([t['num_buys'] for t in trades]),
        'win_rate': len([t for t in trades if t['return'] > 0]) / len(trades) * 100
    }


def optimize_ticker(ticker, df):
    """티커별 최적화"""
    print(f"\n{'='*60}")
    print(f"🔍 {ticker} 최적화 시작")
    print(f"{'='*60}")
    print(f"데이터: {df.index[0].date()} ~ {df.index[-1].date()} ({len(df)}일)")
    
    # 파라미터 범위
    oversold_range = [25, 30, 35, 40, 45]
    buy_exit_range = [35, 40, 45, 50, 55, 60]
    overbought_range = [55, 60, 65, 70, 75, 80]
    sell_exit_range = [40, 45, 50, 55, 60]
    
    results = []
    
    for oversold, buy_exit, overbought, sell_exit in product(
        oversold_range, buy_exit_range, overbought_range, sell_exit_range
    ):
        # 유효성 검사
        if buy_exit <= oversold:
            continue
        if sell_exit >= overbought:
            continue
        
        result = backtest_strategy(df, oversold, buy_exit, overbought, sell_exit)
        
        if result['trades'] >= 5:  # 최소 5회 거래
            results.append({
                'oversold': oversold,
                'buy_exit': buy_exit,
                'overbought': overbought,
                'sell_exit': sell_exit,
                **result
            })
    
    if not results:
        print(f"❌ {ticker}: 조건 만족하는 조합 없음")
        return None
    
    # 정렬 (거래수 * 수익률 기준, 물타기 적은 것 선호)
    results.sort(key=lambda x: (
        x['trades'] * x['total_return'] / max(x['max_water'], 1),
    ), reverse=True)
    
    print(f"\n📊 {ticker} TOP 10 조합:")
    print("-" * 100)
    print(f"{'순위':^4} {'매수조건':^12} {'매도조건':^12} {'거래수':^6} {'수익률':^10} {'최대물타기':^8} {'평균물타기':^8} {'승률':^6}")
    print("-" * 100)
    
    for i, r in enumerate(results[:10], 1):
        print(f"{i:^4} {r['oversold']}/{r['buy_exit']:>2}→{r['buy_exit']:>2} "
              f"{r['overbought']:>2}/{r['sell_exit']:>2}→{r['sell_exit']:>2} "
              f"{r['trades']:^6} {r['total_return']:>+8.1f}% "
              f"{r['max_water']:^8} {r['avg_water']:^8.1f} {r['win_rate']:>5.0f}%")
    
    best = results[0]
    print(f"\n✅ {ticker} 최적 조합:")
    print(f"   매수: RSI < {best['oversold']} → ≥ {best['buy_exit']}")
    print(f"   매도: RSI > {best['overbought']} → ≤ {best['sell_exit']}")
    print(f"   거래: {best['trades']}회, 수익률: {best['total_return']:+.1f}%, 최대 물타기: {best['max_water']}회")
    
    return best


def main():
    print("=" * 60)
    print("🏪 WMT, GLD 최적화 (Wilder's Smoothing RSI)")
    print("=" * 60)
    
    tickers = ['WMT', 'GLD']
    
    # 데이터 로드
    data = {}
    for ticker in tickers:
        print(f"\n📥 {ticker} 데이터 로드 중...")
        df = yf.download(ticker, period='10y', progress=False, auto_adjust=False)
        
        # MultiIndex 처리
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if len(df) > 0:
            data[ticker] = df
            print(f"   ✅ {len(df)}일 데이터 로드")
        else:
            print(f"   ❌ 데이터 로드 실패")
    
    # 최적화
    optimized = {}
    for ticker in tickers:
        if ticker in data:
            best = optimize_ticker(ticker, data[ticker])
            if best:
                optimized[ticker] = best
    
    # AAPL 전략으로도 테스트
    print("\n" + "=" * 60)
    print("📊 AAPL 전략 (35/40→60/45) 적용 결과")
    print("=" * 60)
    
    aapl_strategy = {
        'oversold': 35,
        'buy_exit': 40,
        'overbought': 60,
        'sell_exit': 45
    }
    
    for ticker in tickers:
        if ticker in data:
            result = backtest_strategy(
                data[ticker],
                aapl_strategy['oversold'],
                aapl_strategy['buy_exit'],
                aapl_strategy['overbought'],
                aapl_strategy['sell_exit']
            )
            print(f"\n{ticker} (AAPL 전략):")
            print(f"   거래: {result['trades']}회")
            print(f"   수익률: {result['total_return']:+.1f}%")
            print(f"   최대 물타기: {result['max_water']}회")
            print(f"   승률: {result['win_rate']:.0f}%")
    
    # 최종 비교
    print("\n" + "=" * 60)
    print("📊 최종 비교: 최적화 vs AAPL 전략")
    print("=" * 60)
    print(f"\n{'종목':^6} {'전략':^20} {'거래수':^8} {'수익률':^10} {'최대물타기':^10}")
    print("-" * 60)
    
    for ticker in tickers:
        if ticker in data:
            # 최적화 결과
            if ticker in optimized:
                opt = optimized[ticker]
                print(f"{ticker:^6} 최적화 {opt['oversold']}/{opt['buy_exit']}→{opt['overbought']}/{opt['sell_exit']} "
                      f"{opt['trades']:^8} {opt['total_return']:>+8.1f}% {opt['max_water']:^10}")
            
            # AAPL 전략
            aapl_result = backtest_strategy(
                data[ticker],
                aapl_strategy['oversold'],
                aapl_strategy['buy_exit'],
                aapl_strategy['overbought'],
                aapl_strategy['sell_exit']
            )
            print(f"{ticker:^6} AAPL (35/40→60/45) "
                  f"{aapl_result['trades']:^8} {aapl_result['total_return']:>+8.1f}% {aapl_result['max_water']:^10}")


if __name__ == '__main__':
    main()

