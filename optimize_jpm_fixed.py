"""
JPM 파라미터 재최적화
- 10년 데이터 기준
- 실제 금액 기준 수익률 (물타기 시 투자금 증가)
- ⚠️ 최소 거래 기준 추가: 연 0.8회 이상, 총 8회 이상!
"""

import yfinance as yf
import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

TICKER = 'JPM'

# 파라미터 탐색 범위 (더 넓게)
RSI_OVERSOLD_RANGE = [25, 30, 35, 40, 45]      # 과매도 기준
RSI_BUY_EXIT_RANGE = [30, 35, 40, 45, 50, 55]  # 매수 탈출 기준
RSI_OVERBOUGHT_RANGE = [65, 70, 75, 80, 85]    # 과매수 기준
RSI_SELL_EXIT_RANGE = [40, 45, 50, 55, 60]     # 매도 탈출 기준

CAPITAL_PER_ENTRY = 1000

# ⚠️ 최소 거래 기준!
MIN_TOTAL_TRADES = 8       # 최소 8번 거래
MIN_TRADES_PER_YEAR = 0.8  # 연 최소 0.8회


def calculate_rsi(prices: pd.Series, period: int = 14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def load_data():
    print(f"⏳ {TICKER} 데이터 로딩...")
    df = yf.download(TICKER, period='10y', progress=False)
    
    if df.empty:
        return None
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df['rsi'] = calculate_rsi(df['Close'])
    
    print(f"✅ {len(df)}일 데이터 ({df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')})")
    return df


def simulate_strategy(df: pd.DataFrame, params: dict):
    """전략 시뮬레이션 (실제 금액 기준)"""
    rsi_oversold = params['rsi_oversold']
    rsi_buy_exit = params['rsi_buy_exit']
    rsi_overbought = params['rsi_overbought']
    rsi_sell_exit = params['rsi_sell_exit']
    
    if rsi_buy_exit <= rsi_oversold:
        return None
    if rsi_sell_exit >= rsi_overbought:
        return None
    
    # 매수 시그널
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
    
    # 매도 시그널
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
    
    # 거래 시뮬레이션
    all_buy_dates = {bs['confirm_date']: bs for bs in buy_signals}
    all_sell_dates = {ss['confirm_date']: ss for ss in sell_signals}
    
    trades = []
    positions = []
    max_drawdown = 0
    
    for idx in range(len(df)):
        current_date = df.index[idx]
        current_price = df['Close'].iloc[idx]
        
        if positions:
            n = len(positions)
            total_invested = n * CAPITAL_PER_ENTRY
            total_quantity = sum(CAPITAL_PER_ENTRY / p['price'] for p in positions)
            avg_price = total_invested / total_quantity
            
            current_return = (current_price / avg_price - 1) * 100
            if current_return < max_drawdown:
                max_drawdown = current_return
            
            if current_date in all_sell_dates:
                sell_price = all_sell_dates[current_date]['confirm_price']
                sell_return = (sell_price / avg_price - 1) * 100
                
                if sell_return > 0:  # profit_only
                    profit = total_invested * sell_return / 100
                    trades.append({
                        'entry_date': positions[0]['date'],
                        'exit_date': current_date,
                        'num_buys': n,
                        'invested': total_invested,
                        'profit': profit,
                        'return': sell_return,
                    })
                    positions = []
        
        if current_date in all_buy_dates:
            positions.append({
                'date': current_date,
                'price': all_buy_dates[current_date]['confirm_price']
            })
    
    if not trades:
        return None
    
    # 결과 계산
    total_trades = len(trades)
    
    # 연간 거래 횟수
    first_trade = trades[0]['entry_date']
    last_trade = trades[-1]['exit_date']
    years = (last_trade - first_trade).days / 365
    trades_per_year = total_trades / years if years > 0 else 0
    
    # ⚠️ 최소 거래 기준 체크!
    if total_trades < MIN_TOTAL_TRADES:
        return None
    if trades_per_year < MIN_TRADES_PER_YEAR:
        return None
    
    wins = len([t for t in trades if t['return'] > 0])
    total_invested = sum(t['invested'] for t in trades)
    total_profit = sum(t['profit'] for t in trades)
    total_return = (total_profit / total_invested * 100) if total_invested > 0 else 0
    
    avg_buys = np.mean([t['num_buys'] for t in trades])
    max_buys = max([t['num_buys'] for t in trades])
    
    return {
        'total_trades': total_trades,
        'win_rate': wins / total_trades * 100,
        'total_invested': total_invested,
        'total_profit': total_profit,
        'total_return': total_return,
        'avg_buys': avg_buys,
        'max_buys': max_buys,
        'max_drawdown': max_drawdown,
        'trades_per_year': trades_per_year,
        'current_water': len(positions),
        'trades': trades,
        'buy_signals': buy_signals
    }


def calculate_score(result: dict):
    """점수 계산 - 거래수 더 중요!"""
    if result is None:
        return -999
    
    # 1. 수익률 점수 (30점)
    return_score = min(30, max(0, result['total_return'] * 1.2))
    
    # 2. 거래 횟수 점수 (30점) - 더 중요하게!
    if 1.0 <= result['trades_per_year'] <= 2.0:
        trade_score = 30
    elif 0.8 <= result['trades_per_year'] <= 2.5:
        trade_score = 25
    elif 0.5 <= result['trades_per_year'] <= 3.0:
        trade_score = 15
    else:
        trade_score = 5
    
    # 3. 물타기 점수 (20점)
    if result['avg_buys'] <= 2:
        water_score = 20
    elif result['avg_buys'] <= 3:
        water_score = 15
    elif result['avg_buys'] <= 4:
        water_score = 10
    else:
        water_score = 5
    
    # 4. 최대 물타기 점수 (10점)
    if result['max_buys'] <= 4:
        max_water_score = 10
    elif result['max_buys'] <= 6:
        max_water_score = 7
    else:
        max_water_score = 3
    
    # 5. 승률 점수 (10점)
    winrate_score = result['win_rate'] / 10
    
    return return_score + trade_score + water_score + max_water_score + winrate_score


def main():
    print("="*80)
    print(f"🔧 {TICKER} 파라미터 재최적화")
    print("="*80)
    print(f"⚠️ 최소 기준: 총 {MIN_TOTAL_TRADES}회 이상, 연 {MIN_TRADES_PER_YEAR}회 이상!")
    print(f"투자 단위: ${CAPITAL_PER_ENTRY:,}/회")
    
    df = load_data()
    if df is None:
        print("데이터 로드 실패!")
        return
    
    # RSI 과매도 빈도 확인
    rsi_stats = {}
    for threshold in [25, 30, 35, 40, 45]:
        count = (df['rsi'] < threshold).sum()
        pct = count / len(df) * 100
        rsi_stats[threshold] = {'count': count, 'pct': pct}
    
    print(f"\n📊 RSI 과매도 빈도 (10년간)")
    print("-"*40)
    for t, s in rsi_stats.items():
        print(f"  RSI < {t}: {s['count']}회 ({s['pct']:.1f}%)")
    
    # 파라미터 최적화
    results = []
    total_combinations = (len(RSI_OVERSOLD_RANGE) * len(RSI_BUY_EXIT_RANGE) * 
                          len(RSI_OVERBOUGHT_RANGE) * len(RSI_SELL_EXIT_RANGE))
    
    print(f"\n⏳ {total_combinations}개 조합 테스트 중...")
    
    valid_count = 0
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
            valid_count += 1
            score = calculate_score(result)
            results.append({
                'params': params,
                'result': result,
                'score': score
            })
    
    print(f"✅ 유효한 조합: {valid_count}개 (거래 기준 충족)")
    
    if not results:
        print("❌ 유효한 결과 없음!")
        return
    
    # 점수순 정렬
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # TOP 15 출력
    print(f"\n📊 TOP 15 파라미터 조합")
    print("-"*100)
    print(f"{'순위':<4} {'RSI설정':^22} {'수익률':>10} {'거래수':>8} {'연거래':>8} {'평균물타기':>10} {'최대물타기':>10} {'점수':>8}")
    print("-"*100)
    
    for i, r in enumerate(results[:15]):
        p = r['params']
        res = r['result']
        rsi_str = f"{p['rsi_oversold']}/{p['rsi_buy_exit']}→{p['rsi_overbought']}/{p['rsi_sell_exit']}"
        print(f"{i+1:<4} {rsi_str:^22} {res['total_return']:>+9.1f}% {res['total_trades']:>7}회 "
              f"{res['trades_per_year']:>7.1f}회 {res['avg_buys']:>9.1f}회 {res['max_buys']:>9}회 {r['score']:>7.1f}")
    
    # 최적 파라미터
    best = results[0]
    p = best['params']
    r = best['result']
    
    print(f"\n{'='*80}")
    print(f"🏆 최적 파라미터")
    print(f"{'='*80}")
    print(f"\n  📈 전략: RSI {p['rsi_oversold']}/{p['rsi_buy_exit']} → {p['rsi_overbought']}/{p['rsi_sell_exit']}")
    print(f"  💰 총 수익률: {r['total_return']:+.1f}%")
    print(f"  📊 거래 횟수: {r['total_trades']}회 (연 {r['trades_per_year']:.1f}회)")
    print(f"  💧 물타기: 평균 {r['avg_buys']:.1f}회, 최대 {r['max_buys']}회")
    print(f"  📉 최대 손실: {r['max_drawdown']:.1f}%")
    print(f"  ✅ 승률: {r['win_rate']:.0f}%")
    
    # 매수 시그널 날짜
    print(f"\n📅 매수 시그널 날짜 ({len(r['buy_signals'])}개)")
    print("-"*50)
    for bs in r['buy_signals'][-15:]:  # 최근 15개
        print(f"  {bs['confirm_date'].strftime('%Y-%m-%d')}: ${bs['confirm_price']:.2f}")
    
    # 거래 내역
    print(f"\n💹 거래 내역 ({r['total_trades']}개)")
    print("-"*80)
    print(f"{'기간':^28} {'물타기':>8} {'투자금':>12} {'손익':>12} {'수익률':>10}")
    print("-"*80)
    for t in r['trades']:
        period = f"{t['entry_date'].strftime('%Y-%m-%d')} ~ {t['exit_date'].strftime('%Y-%m-%d')}"
        print(f"{period:^28} {t['num_buys']:>7}회 ${t['invested']:>10,} ${t['profit']:>+10,.0f} {t['return']:>+9.1f}%")
    
    # 대시보드 설정 안내
    print(f"\n{'='*80}")
    print(f"📝 대시보드 설정 업데이트")
    print(f"{'='*80}")
    print(f"""
dashboard_jpm.py 파라미터 수정:

RSI_OVERSOLD = {p['rsi_oversold']}
RSI_BUY_EXIT = {p['rsi_buy_exit']}
RSI_OVERBOUGHT = {p['rsi_overbought']}
RSI_SELL_EXIT = {p['rsi_sell_exit']}
""")


if __name__ == "__main__":
    main()

