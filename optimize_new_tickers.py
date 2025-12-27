"""
XOM, XLE, JPM 파라미터 최적화
- 10년 데이터 기준
- 실제 금액 기준 수익률 (물타기 시 투자금 증가)
- 최적화 기준: 수익률 / 거래수 / 리스크(물타기)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 최적화 대상 종목
TICKERS = ['XOM', 'XLE', 'JPM']

# 파라미터 탐색 범위
RSI_OVERSOLD_RANGE = [25, 30, 35, 40]      # 과매도 기준
RSI_BUY_EXIT_RANGE = [30, 35, 40, 45, 50]  # 매수 탈출 기준
RSI_OVERBOUGHT_RANGE = [70, 75, 80, 85]    # 과매수 기준
RSI_SELL_EXIT_RANGE = [40, 45, 50, 55, 60] # 매도 탈출 기준

CAPITAL_PER_ENTRY = 1000  # 매수마다 $1,000


def calculate_rsi(prices: pd.Series, period: int = 14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def load_data(ticker: str):
    print(f"  ⏳ {ticker} 데이터 로딩...")
    df = yf.download(ticker, period='10y', progress=False)
    
    if df.empty:
        return None
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df['rsi'] = calculate_rsi(df['Close'])
    
    print(f"  ✅ {len(df)}일 데이터 ({df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')})")
    return df


def simulate_strategy(df: pd.DataFrame, params: dict):
    """
    전략 시뮬레이션 (실제 금액 기준!)
    물타기마다 투자금 $1,000씩 증가
    """
    rsi_oversold = params['rsi_oversold']
    rsi_buy_exit = params['rsi_buy_exit']
    rsi_overbought = params['rsi_overbought']
    rsi_sell_exit = params['rsi_sell_exit']
    
    # 파라미터 유효성 체크
    if rsi_buy_exit <= rsi_oversold:
        return None
    if rsi_sell_exit >= rsi_overbought:
        return None
    
    # 매수 시그널 찾기
    buy_signals = []
    in_oversold = False
    last_signal_date = None
    
    for idx in range(len(df)):
        rsi = df['rsi'].iloc[idx]
        if pd.isna(rsi):
            continue
        
        if rsi < rsi_oversold:
            in_oversold = True
            last_signal_date = df.index[idx]
        else:
            if in_oversold and rsi >= rsi_buy_exit and last_signal_date is not None:
                buy_signals.append({
                    'confirm_date': df.index[idx],
                    'confirm_price': df['Close'].iloc[idx],
                })
                in_oversold = False
                last_signal_date = None
    
    # 매도 시그널 찾기
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
            if in_overbought and rsi <= rsi_sell_exit and last_signal_date is not None:
                sell_signals.append({
                    'confirm_date': df.index[idx],
                    'confirm_price': df['Close'].iloc[idx]
                })
                in_overbought = False
                last_signal_date = None
    
    # 거래 시뮬레이션 (실제 금액 기준)
    all_buy_dates = {bs['confirm_date']: bs for bs in buy_signals}
    all_sell_dates = {ss['confirm_date']: ss for ss in sell_signals}
    
    trades = []
    positions = []
    max_drawdown = 0
    
    for idx in range(len(df)):
        current_date = df.index[idx]
        current_price = df['Close'].iloc[idx]
        
        if positions:
            # 실제 금액 기준 평균가 계산
            n = len(positions)
            total_invested = n * CAPITAL_PER_ENTRY
            total_quantity = sum(CAPITAL_PER_ENTRY / p['price'] for p in positions)
            avg_price = total_invested / total_quantity
            
            current_return = (current_price / avg_price - 1) * 100
            
            if current_return < max_drawdown:
                max_drawdown = current_return
            
            # 매도 조건: RSI 시그널 + 수익일 때만 (profit_only)
            if current_date in all_sell_dates:
                sell_price = all_sell_dates[current_date]['confirm_price']
                sell_return = (sell_price / avg_price - 1) * 100
                
                if sell_return > 0:  # 수익일 때만 익절!
                    # 실제 금액 기준 손익 계산
                    profit = total_invested * sell_return / 100
                    
                    trades.append({
                        'entry_date': positions[0]['date'],
                        'exit_date': current_date,
                        'num_buys': n,
                        'invested': total_invested,
                        'profit': profit,
                        'return': sell_return,
                        'holding_days': (current_date - positions[0]['date']).days
                    })
                    positions = []
        
        if current_date in all_buy_dates:
            positions.append({
                'date': current_date,
                'price': all_buy_dates[current_date]['confirm_price']
            })
    
    if not trades:
        return None
    
    # 결과 계산 (실제 금액 기준!)
    total_trades = len(trades)
    wins = len([t for t in trades if t['return'] > 0])
    
    # 실제 금액 기준 총 투자금 & 총 손익
    total_invested = sum(t['invested'] for t in trades)
    total_profit = sum(t['profit'] for t in trades)
    total_return = (total_profit / total_invested * 100) if total_invested > 0 else 0
    
    # 물타기 통계
    avg_buys = np.mean([t['num_buys'] for t in trades])
    max_buys = max([t['num_buys'] for t in trades])
    
    # 연간 거래 횟수
    first_trade = trades[0]['entry_date']
    last_trade = trades[-1]['exit_date']
    years = (last_trade - first_trade).days / 365
    trades_per_year = total_trades / years if years > 0 else 0
    
    # 보유 기간
    avg_holding = np.mean([t['holding_days'] for t in trades])
    max_holding = max([t['holding_days'] for t in trades])
    
    # 현재 보유 중 체크
    current_water = len(positions)
    
    return {
        'total_trades': total_trades,
        'win_rate': wins / total_trades * 100,
        'total_invested': total_invested,
        'total_profit': total_profit,
        'total_return': total_return,  # 실제 금액 기준 수익률!
        'avg_buys': avg_buys,
        'max_buys': max_buys,
        'max_drawdown': max_drawdown,
        'trades_per_year': trades_per_year,
        'avg_holding': avg_holding,
        'max_holding': max_holding,
        'current_water': current_water,
        'trades': trades
    }


def calculate_score(result: dict):
    """
    종합 점수 계산
    기준: 수익률 / 거래수 / 리스크(물타기)
    """
    if result is None:
        return -999
    
    # 1. 수익률 점수 (40점) - 실제 금액 기준!
    return_score = min(40, max(0, result['total_return'] * 1.5))
    
    # 2. 거래 횟수 점수 (20점) - 연 1~3회가 이상적
    if 1 <= result['trades_per_year'] <= 3:
        trade_score = 20
    elif 0.5 <= result['trades_per_year'] <= 4:
        trade_score = 15
    else:
        trade_score = 5
    
    # 3. 물타기 점수 (20점) - 적을수록 좋음
    if result['avg_buys'] <= 2:
        water_score = 20
    elif result['avg_buys'] <= 3:
        water_score = 15
    elif result['avg_buys'] <= 4:
        water_score = 10
    else:
        water_score = max(0, 20 - result['avg_buys'] * 3)
    
    # 4. 최대 물타기 패널티 (10점)
    if result['max_buys'] <= 3:
        max_water_score = 10
    elif result['max_buys'] <= 5:
        max_water_score = 7
    elif result['max_buys'] <= 8:
        max_water_score = 4
    else:
        max_water_score = 0
    
    # 5. 승률 점수 (10점)
    winrate_score = result['win_rate'] / 10
    
    total_score = return_score + trade_score + water_score + max_water_score + winrate_score
    
    return total_score


def optimize_ticker(ticker: str, df: pd.DataFrame):
    """종목별 파라미터 최적화"""
    print(f"\n{'='*60}")
    print(f"🔧 {ticker} 파라미터 최적화")
    print(f"{'='*60}")
    
    results = []
    total_combinations = (len(RSI_OVERSOLD_RANGE) * len(RSI_BUY_EXIT_RANGE) * 
                          len(RSI_OVERBOUGHT_RANGE) * len(RSI_SELL_EXIT_RANGE))
    
    print(f"  총 {total_combinations}개 조합 테스트 중...")
    
    for oversold, buy_exit, overbought, sell_exit in product(
        RSI_OVERSOLD_RANGE, RSI_BUY_EXIT_RANGE, RSI_OVERBOUGHT_RANGE, RSI_SELL_EXIT_RANGE
    ):
        params = {
            'rsi_oversold': oversold,
            'rsi_buy_exit': buy_exit,
            'rsi_overbought': overbought,
            'rsi_sell_exit': sell_exit
        }
        
        result = simulate_strategy(df, params)
        
        if result:
            score = calculate_score(result)
            results.append({
                'params': params,
                'result': result,
                'score': score
            })
    
    if not results:
        print(f"  ❌ 유효한 결과 없음")
        return None
    
    # 점수순 정렬
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # TOP 10 출력
    print(f"\n  📊 TOP 10 파라미터 조합")
    print("  " + "-"*80)
    print(f"  {'순위':<4} {'RSI설정':^20} {'수익률':>10} {'거래수':>8} {'평균물타기':>10} {'최대물타기':>10} {'점수':>8}")
    print("  " + "-"*80)
    
    for i, r in enumerate(results[:10]):
        p = r['params']
        res = r['result']
        rsi_str = f"{p['rsi_oversold']}/{p['rsi_buy_exit']}→{p['rsi_overbought']}/{p['rsi_sell_exit']}"
        print(f"  {i+1:<4} {rsi_str:^20} {res['total_return']:>+9.1f}% {res['total_trades']:>7}회 "
              f"{res['avg_buys']:>9.1f}회 {res['max_buys']:>9}회 {r['score']:>7.1f}")
    
    # 최적 파라미터
    best = results[0]
    
    print(f"\n  🏆 최적 파라미터: RSI {best['params']['rsi_oversold']}/{best['params']['rsi_buy_exit']} → "
          f"{best['params']['rsi_overbought']}/{best['params']['rsi_sell_exit']}")
    
    return best


def print_final_comparison(all_results: dict):
    """최종 비교"""
    print("\n" + "="*80)
    print("📊 최적화 완료 - 최종 비교")
    print("="*80)
    
    # 기존 종목 최적 파라미터 (참고용)
    existing = {
        'QQQ': {'params': '35/40→75/50', 'return': 20.8, 'trades': 10, 'avg_buys': 4.0, 'max_buys': 10},
        'AAPL': {'params': '30/35→75/50', 'return': 28.1, 'trades': 10, 'avg_buys': 1.9, 'max_buys': 3},
        'SMH': {'params': '35/40→75/45', 'return': 33.5, 'trades': 10, 'avg_buys': 2.9, 'max_buys': 8},
    }
    
    print("\n### 기존 종목 (참고)")
    print("-"*80)
    print(f"{'종목':<8} {'파라미터':^20} {'총수익률':>10} {'거래수':>8} {'평균물타기':>10} {'최대물타기':>10}")
    print("-"*80)
    for ticker, data in existing.items():
        print(f"{ticker:<8} {data['params']:^20} {data['return']:>+9.1f}% {data['trades']:>7}회 "
              f"{data['avg_buys']:>9.1f}회 {data['max_buys']:>9}회")
    
    print("\n### 새 종목 (최적화 결과)")
    print("-"*80)
    print(f"{'종목':<8} {'파라미터':^20} {'총수익률':>10} {'거래수':>8} {'평균물타기':>10} {'최대물타기':>10} {'점수':>8}")
    print("-"*80)
    
    for ticker, best in all_results.items():
        if best:
            p = best['params']
            r = best['result']
            params_str = f"{p['rsi_oversold']}/{p['rsi_buy_exit']}→{p['rsi_overbought']}/{p['rsi_sell_exit']}"
            print(f"{ticker:<8} {params_str:^20} {r['total_return']:>+9.1f}% {r['total_trades']:>7}회 "
                  f"{r['avg_buys']:>9.1f}회 {r['max_buys']:>9}회 {best['score']:>7.1f}")
    
    # 상세 분석
    print("\n" + "="*80)
    print("📋 상세 분석")
    print("="*80)
    
    for ticker, best in all_results.items():
        if best:
            p = best['params']
            r = best['result']
            
            print(f"\n【{ticker}】")
            print(f"  📈 최적 전략: RSI {p['rsi_oversold']}/{p['rsi_buy_exit']} → {p['rsi_overbought']}/{p['rsi_sell_exit']}")
            print(f"  💰 총 수익률: {r['total_return']:+.1f}% (실제 금액 기준)")
            print(f"  💵 총 투자금: ${r['total_invested']:,.0f}")
            print(f"  💵 총 손익: ${r['total_profit']:+,.0f}")
            print(f"  📊 거래 횟수: {r['total_trades']}회 (연 {r['trades_per_year']:.1f}회)")
            print(f"  💧 물타기: 평균 {r['avg_buys']:.1f}회, 최대 {r['max_buys']}회")
            print(f"  📅 보유기간: 평균 {r['avg_holding']:.0f}일, 최대 {r['max_holding']}일")
            print(f"  📉 최대 손실: {r['max_drawdown']:.1f}%")
            print(f"  🎯 현재 물타기 중: {r['current_water']}회")
            
            # 최근 5개 거래
            print(f"\n  최근 거래 내역:")
            for t in r['trades'][-5:]:
                print(f"    {t['entry_date'].strftime('%Y-%m-%d')} ~ {t['exit_date'].strftime('%Y-%m-%d')}: "
                      f"{t['num_buys']}회 물타기, ${t['invested']:,} → ${t['profit']:+,.0f} ({t['return']:+.1f}%)")


def main():
    print("="*80)
    print("🔧 XOM, XLE, JPM 파라미터 최적화")
    print("="*80)
    print("기준: 수익률(실제금액) / 거래수 / 물타기 리스크")
    print(f"투자 단위: ${CAPITAL_PER_ENTRY:,}/회")
    
    all_results = {}
    
    for ticker in TICKERS:
        df = load_data(ticker)
        if df is not None:
            best = optimize_ticker(ticker, df)
            all_results[ticker] = best
    
    print_final_comparison(all_results)
    
    # 추천
    print("\n" + "="*80)
    print("💡 최종 추천")
    print("="*80)
    
    valid_results = [(t, r) for t, r in all_results.items() if r]
    if valid_results:
        # 점수순 정렬
        valid_results.sort(key=lambda x: x[1]['score'], reverse=True)
        
        print("\n새 종목 순위 (최적화 후):")
        for i, (ticker, best) in enumerate(valid_results):
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."
            r = best['result']
            print(f"  {medal} {ticker}: 점수 {best['score']:.1f}, "
                  f"수익률 {r['total_return']:+.1f}%, 물타기 평균 {r['avg_buys']:.1f}회")


if __name__ == "__main__":
    main()

