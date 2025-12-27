"""
QQQ, AAPL, SMH 파라미터 최적화 - 거래 수 늘리기 버전
- 현재 전략과 비교
- 수익률 유지하면서 거래 수 늘리기
"""

import sys
sys.path.insert(0, '.')

from src.data.cache import DataCache
from src.data.fetcher import DataFetcher
from src.data.validator import DataValidator
from src.features.technical import TechnicalIndicators
from src.utils.helpers import load_config
import pandas as pd
import numpy as np
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# 현재 전략
CURRENT_STRATEGIES = {
    'QQQ': {'rsi_oversold': 35, 'rsi_buy_exit': 40, 'rsi_overbought': 75, 'rsi_sell_exit': 50},
    'AAPL': {'rsi_oversold': 30, 'rsi_buy_exit': 35, 'rsi_overbought': 75, 'rsi_sell_exit': 50},
    'SMH': {'rsi_oversold': 35, 'rsi_buy_exit': 40, 'rsi_overbought': 75, 'rsi_sell_exit': 45},
}

# 파라미터 탐색 범위 (거래 늘리기 위해)
RSI_OVERSOLD_RANGE = [30, 35, 40, 45]
RSI_BUY_EXIT_RANGE = [35, 40, 45, 50, 55]
RSI_OVERBOUGHT_RANGE = [60, 65, 70, 75]
RSI_SELL_EXIT_RANGE = [45, 50, 55, 60]

CAPITAL_PER_ENTRY = 1000

# 거래 기준
MIN_TOTAL_TRADES = 10
MIN_TRADES_PER_YEAR = 1.0


def load_data(ticker: str):
    config = load_config()
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


def simulate_strategy(df: pd.DataFrame, params: dict):
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
                
                if sell_return > 0:
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
    
    total_trades = len(trades)
    
    first_trade = trades[0]['entry_date']
    last_trade = trades[-1]['exit_date']
    years = (last_trade - first_trade).days / 365
    trades_per_year = total_trades / years if years > 0 else 0
    
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


def calculate_score(result: dict, current_return: float):
    """점수 계산 - 거래 수 중심, 수익률 유지"""
    if result is None:
        return -999
    
    # 수익률이 현재보다 50% 이상 떨어지면 탈락
    if result['total_return'] < current_return * 0.5:
        return -999
    
    # 1. 거래 횟수 점수 (40점)
    if result['trades_per_year'] >= 1.5:
        trade_score = 40
    elif result['trades_per_year'] >= 1.2:
        trade_score = 30
    elif result['trades_per_year'] >= 1.0:
        trade_score = 20
    else:
        trade_score = 10
    
    # 2. 수익률 점수 (25점)
    return_score = min(25, max(0, result['total_return'] * 0.8))
    
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
        max_water_score = 6
    else:
        max_water_score = 2
    
    # 5. 승률 점수 (5점)
    winrate_score = result['win_rate'] / 20
    
    return trade_score + return_score + water_score + max_water_score + winrate_score


def optimize_ticker(ticker: str, df: pd.DataFrame, current_params: dict):
    """종목별 최적화"""
    print(f"\n{'='*80}")
    print(f"🔧 {ticker} 최적화")
    print(f"{'='*80}")
    
    # 현재 전략 결과
    current_result = simulate_strategy(df, current_params)
    
    if current_result:
        p = current_params
        print(f"\n📊 현재 전략: RSI {p['rsi_oversold']}/{p['rsi_buy_exit']}→{p['rsi_overbought']}/{p['rsi_sell_exit']}")
        print(f"   거래: {current_result['total_trades']}회 (연 {current_result['trades_per_year']:.1f}회)")
        print(f"   수익률: {current_result['total_return']:+.1f}%")
        print(f"   물타기: 평균 {current_result['avg_buys']:.1f}회, 최대 {current_result['max_buys']}회")
        current_return = current_result['total_return']
    else:
        print("   ❌ 현재 전략 결과 없음")
        current_return = 0
    
    # 새 조합 탐색
    results = []
    
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
        
        if result and result['total_trades'] >= MIN_TOTAL_TRADES:
            score = calculate_score(result, current_return)
            if score > 0:
                results.append({
                    'params': params,
                    'result': result,
                    'score': score
                })
    
    if not results:
        print("   ❌ 조건 충족 조합 없음")
        return None, current_result
    
    # 정렬
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # TOP 10
    print(f"\n📊 TOP 10 (거래 수 늘린 조합)")
    print("-"*100)
    print(f"{'순위':<4} {'RSI설정':^22} {'수익률':>10} {'거래수':>8} {'연거래':>8} {'평균물타기':>10} {'최대물타기':>10}")
    print("-"*100)
    
    for i, r in enumerate(results[:10]):
        p = r['params']
        res = r['result']
        rsi_str = f"{p['rsi_oversold']}/{p['rsi_buy_exit']}→{p['rsi_overbought']}/{p['rsi_sell_exit']}"
        print(f"{i+1:<4} {rsi_str:^22} {res['total_return']:>+9.1f}% {res['total_trades']:>7}회 "
              f"{res['trades_per_year']:>7.1f}회 {res['avg_buys']:>9.1f}회 {res['max_buys']:>9}회")
    
    return results[0], current_result


def main():
    print("="*80)
    print("🔧 QQQ, AAPL, SMH 최적화 - 거래 수 늘리기")
    print("="*80)
    
    all_comparisons = {}
    
    for ticker, current_params in CURRENT_STRATEGIES.items():
        print(f"\n⏳ {ticker} 데이터 로딩...")
        df = load_data(ticker)
        print(f"   ✅ {len(df)}일")
        
        best, current = optimize_ticker(ticker, df, current_params)
        all_comparisons[ticker] = {
            'current': current,
            'current_params': current_params,
            'best': best
        }
    
    # 최종 비교
    print("\n" + "="*80)
    print("📊 최종 비교: 현재 vs 거래 늘린 조합")
    print("="*80)
    
    for ticker, data in all_comparisons.items():
        current = data['current']
        cp = data['current_params']
        best = data['best']
        
        print(f"\n【{ticker}】")
        print("-"*70)
        
        if current and best:
            bp = best['params']
            br = best['result']
            
            # 변화율 계산
            trade_change = (br['trades_per_year'] / current['trades_per_year'] - 1) * 100 if current['trades_per_year'] > 0 else 0
            return_change = (br['total_return'] / current['total_return'] - 1) * 100 if current['total_return'] > 0 else 0
            
            curr_param_str = f"{cp['rsi_oversold']}/{cp['rsi_buy_exit']}→{cp['rsi_overbought']}/{cp['rsi_sell_exit']}"
            best_param_str = f"{bp['rsi_oversold']}/{bp['rsi_buy_exit']}→{bp['rsi_overbought']}/{bp['rsi_sell_exit']}"
            
            print(f"{'':^20} {'현재':^20} {'거래 늘린 조합':^20} {'변화':^12}")
            print("-"*70)
            print(f"{'파라미터':^20} {curr_param_str:^20} {best_param_str:^20}")
            print(f"{'거래 수':^20} {current['total_trades']}회 (연{current['trades_per_year']:.1f}):^20 "
                  f"{br['total_trades']}회 (연{br['trades_per_year']:.1f}):^20 {trade_change:+.0f}%")
            print(f"{'수익률':^20} {current['total_return']:+.1f}%:^20 {br['total_return']:+.1f}%:^20 {return_change:+.0f}%")
            print(f"{'평균 물타기':^20} {current['avg_buys']:.1f}회:^20 {br['avg_buys']:.1f}회")
            print(f"{'최대 물타기':^20} {current['max_buys']}회:^20 {br['max_buys']}회")
            print(f"{'승률':^20} {current['win_rate']:.0f}%:^20 {br['win_rate']:.0f}%")
        else:
            print("   결과 없음")
    
    # 추천
    print("\n" + "="*80)
    print("💡 추천")
    print("="*80)
    
    for ticker, data in all_comparisons.items():
        current = data['current']
        best = data['best']
        
        if current and best:
            br = best['result']
            bp = best['params']
            
            trade_increase = br['trades_per_year'] / current['trades_per_year'] if current['trades_per_year'] > 0 else 1
            return_decrease = 1 - br['total_return'] / current['total_return'] if current['total_return'] > 0 else 0
            
            print(f"\n{ticker}:")
            if trade_increase >= 1.3 and return_decrease <= 0.3:
                print(f"  ✅ 추천: {bp['rsi_oversold']}/{bp['rsi_buy_exit']}→{bp['rsi_overbought']}/{bp['rsi_sell_exit']}")
                print(f"     거래 {trade_increase:.1f}배 증가, 수익률 {return_decrease*100:.0f}% 감소")
            elif trade_increase >= 1.2:
                print(f"  🤔 고려: {bp['rsi_oversold']}/{bp['rsi_buy_exit']}→{bp['rsi_overbought']}/{bp['rsi_sell_exit']}")
                print(f"     거래 {trade_increase:.1f}배 증가, 수익률 {return_decrease*100:.0f}% 감소")
            else:
                print(f"  ⚠️ 현재 유지가 나음")
                print(f"     거래 증가 효과 미미")


if __name__ == "__main__":
    main()

