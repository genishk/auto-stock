"""
대안 종목 (XOM, XLE, JPM) 백테스트
AAPL 전략: RSI 30/35 → 75/50, 골든크로스 OFF, 손절 없음
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 백테스트 대상 종목
TICKERS = ['XOM', 'XLE', 'JPM']

# AAPL 전략 파라미터
STRATEGY = {
    'name': 'AAPL 전략',
    'rsi_oversold': 30,
    'rsi_buy_exit': 35,
    'rsi_overbought': 75,
    'rsi_sell_exit': 50,
    'stop_loss': None,  # 손절 없음
    'capital_per_entry': 1000,
}

# 비교용: QQQ 전략
QQQ_STRATEGY = {
    'name': 'QQQ 전략',
    'rsi_oversold': 35,
    'rsi_buy_exit': 40,
    'rsi_overbought': 75,
    'rsi_sell_exit': 50,
    'stop_loss': None,
    'capital_per_entry': 1000,
}


def calculate_rsi(prices: pd.Series, period: int = 14):
    """RSI 계산"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def load_data(ticker: str, period: str = '10y'):
    """데이터 로드 및 지표 계산"""
    print(f"  ⏳ {ticker} 데이터 로딩...")
    
    df = yf.download(ticker, period=period, progress=False)
    
    if df.empty:
        return None
    
    # MultiIndex 처리
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # RSI 계산
    df['rsi'] = calculate_rsi(df['Close'])
    
    # MA 계산 (골든크로스용 - 참고용)
    df['MA40'] = df['Close'].rolling(window=40).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['golden_cross'] = df['MA40'] > df['MA200']
    
    print(f"  ✅ {ticker}: {len(df)}일 데이터 ({df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')})")
    
    return df


def find_buy_signals(df: pd.DataFrame, params: dict):
    """매수 시그널 찾기"""
    buy_signals = []
    in_oversold = False
    last_signal_date = None
    last_signal_price = None
    
    for idx in range(len(df)):
        rsi = df['rsi'].iloc[idx]
        if pd.isna(rsi):
            continue
        
        if rsi < params['rsi_oversold']:
            in_oversold = True
            last_signal_date = df.index[idx]
            last_signal_price = df['Close'].iloc[idx]
        else:
            if in_oversold and rsi >= params['rsi_buy_exit'] and last_signal_date is not None:
                buy_signals.append({
                    'signal_date': last_signal_date,
                    'confirm_date': df.index[idx],
                    'confirm_price': df['Close'].iloc[idx],
                    'rsi': rsi
                })
                in_oversold = False
                last_signal_date = None
    
    return buy_signals


def find_sell_signals(df: pd.DataFrame, params: dict):
    """매도 시그널 찾기"""
    sell_signals = []
    in_overbought = False
    last_signal_date = None
    
    for idx in range(len(df)):
        rsi = df['rsi'].iloc[idx]
        if pd.isna(rsi):
            continue
        
        if rsi > params['rsi_overbought']:
            in_overbought = True
            last_signal_date = df.index[idx]
        else:
            if in_overbought and rsi <= params['rsi_sell_exit'] and last_signal_date is not None:
                sell_signals.append({
                    'confirm_date': df.index[idx],
                    'confirm_price': df['Close'].iloc[idx]
                })
                in_overbought = False
                last_signal_date = None
    
    return sell_signals


def simulate_trades(df: pd.DataFrame, buy_signals: list, sell_signals: list, params: dict):
    """거래 시뮬레이션 (물타기 + profit_only)"""
    all_buy_dates = {bs['confirm_date']: bs for bs in buy_signals}
    all_sell_dates = {ss['confirm_date']: ss for ss in sell_signals}
    
    trades = []
    positions = []
    capital = params['capital_per_entry']
    stop_loss = params.get('stop_loss')
    
    max_drawdown_ever = 0
    
    for idx in range(len(df)):
        current_date = df.index[idx]
        current_price = df['Close'].iloc[idx]
        
        if positions:
            n = len(positions)
            total_inv = n * capital
            total_qty = sum(capital / p['price'] for p in positions)
            avg_price = total_inv / total_qty
            current_return = (current_price / avg_price - 1) * 100
            
            if current_return < max_drawdown_ever:
                max_drawdown_ever = current_return
            
            exit_reason = None
            exit_price = current_price
            
            # 손절 체크
            if stop_loss is not None and current_return <= stop_loss:
                exit_reason = "손절"
            # RSI 매도 시그널 + 수익인 경우만 익절
            elif current_date in all_sell_dates:
                sell_price = all_sell_dates[current_date]['confirm_price']
                sell_return = (sell_price / avg_price - 1) * 100
                if sell_return > 0:
                    exit_reason = "익절"
                    exit_price = sell_price
            
            if exit_reason:
                final_return = (exit_price / avg_price - 1) * 100
                holding_days = (current_date - positions[0]['date']).days
                
                trades.append({
                    'entry_dates': [p['date'] for p in positions],
                    'entry_prices': [p['price'] for p in positions],
                    'avg_price': avg_price,
                    'num_buys': n,
                    'exit_date': current_date,
                    'exit_price': exit_price,
                    'return': final_return,
                    'exit_reason': exit_reason,
                    'invested': total_inv,
                    'profit': total_inv * final_return / 100,
                    'holding_days': holding_days
                })
                positions = []
        
        if current_date in all_buy_dates:
            positions.append({
                'date': current_date,
                'price': all_buy_dates[current_date]['confirm_price']
            })
    
    # 현재 보유 중
    current_position = None
    if positions:
        n = len(positions)
        total_inv = n * capital
        total_qty = sum(capital / p['price'] for p in positions)
        avg_price = total_inv / total_qty
        current_return = (df['Close'].iloc[-1] / avg_price - 1) * 100
        holding_days = (df.index[-1] - positions[0]['date']).days
        
        current_position = {
            'num_buys': n,
            'invested': total_inv,
            'avg_price': avg_price,
            'unrealized_return': current_return,
            'holding_days': holding_days
        }
    
    return trades, current_position, max_drawdown_ever


def analyze_results(ticker: str, df: pd.DataFrame, trades: list, current_pos: dict, max_dd: float, params: dict):
    """결과 분석"""
    if not trades:
        return None
    
    total_trades = len(trades)
    wins = [t for t in trades if t['return'] > 0]
    losses = [t for t in trades if t['return'] <= 0]
    
    total_invested = sum(t['invested'] for t in trades)
    total_profit = sum(t['profit'] for t in trades)
    
    # 물타기 통계
    avg_buys = np.mean([t['num_buys'] for t in trades])
    max_buys = max([t['num_buys'] for t in trades])
    
    # 보유 기간 통계
    avg_holding = np.mean([t['holding_days'] for t in trades])
    max_holding = max([t['holding_days'] for t in trades])
    
    # 손절 통계
    stoploss_trades = [t for t in trades if t['exit_reason'] == '손절']
    stoploss_loss = sum(t['profit'] for t in stoploss_trades)
    
    # 연간 거래 횟수 (중요!)
    first_trade = trades[0]['entry_dates'][0]
    last_trade = trades[-1]['exit_date']
    total_years = (last_trade - first_trade).days / 365
    trades_per_year = total_trades / total_years if total_years > 0 else total_trades
    
    # 연환산 수익률 (복리)
    if total_years > 0:
        # 단순 연환산
        annual_return = (total_profit / total_invested) / total_years * 100 if total_invested > 0 else 0
        # 복리 계산 (거래당 평균 수익률 × 연간 거래수)
        avg_return_per_trade = np.mean([t['return'] for t in trades])
        compounded_annual = avg_return_per_trade * trades_per_year
    else:
        annual_return = 0
        compounded_annual = 0
    
    # 샤프 비율 근사
    returns = [t['return'] for t in trades]
    sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
    
    return {
        'ticker': ticker,
        'strategy': params['name'],
        'total_trades': total_trades,
        'win_rate': len(wins) / total_trades * 100,
        'total_invested': total_invested,
        'total_profit': total_profit,
        'total_return': total_profit / total_invested * 100 if total_invested > 0 else 0,
        'avg_buys': avg_buys,
        'max_buys': max_buys,
        'avg_holding': avg_holding,
        'max_holding': max_holding,
        'stoploss_count': len(stoploss_trades),
        'stoploss_loss': stoploss_loss,
        'max_drawdown': max_dd,
        'trades_per_year': trades_per_year,
        'annual_return': annual_return,
        'compounded_annual': compounded_annual,
        'sharpe': sharpe,
        'current_position': current_pos,
        'trades': trades,
        'total_years': total_years
    }


def print_results(results: list):
    """결과 출력"""
    print("\n" + "="*80)
    print("📊 백테스트 결과 비교")
    print("="*80)
    
    # 1. 거래 성과 (거래수 포함!)
    print("\n### 1️⃣ 거래 성과")
    print("-" * 80)
    print(f"{'종목':<8} {'거래수':>8} {'연간거래':>10} {'승률':>8} {'총수익률':>10} {'총손익':>12} {'기간':>8}")
    print("-" * 80)
    for r in results:
        print(f"{r['ticker']:<8} {r['total_trades']:>7}회 {r['trades_per_year']:>9.1f}회 "
              f"{r['win_rate']:>7.0f}% {r['total_return']:>+9.1f}% "
              f"${r['total_profit']:>+10,.0f} {r['total_years']:>6.1f}년")
    
    # 2. 복리 효과 분석 (핵심!)
    print("\n### 2️⃣ 복리 효과 분석 ⭐")
    print("-" * 80)
    print(f"{'종목':<8} {'연간거래':>10} {'거래당수익':>12} {'복리연수익':>12} {'10년누적':>14}")
    print("-" * 80)
    for r in results:
        avg_return = np.mean([t['return'] for t in r['trades']]) if r['trades'] else 0
        compounded_10y = ((1 + avg_return/100) ** (r['trades_per_year'] * 10) - 1) * 100
        print(f"{r['ticker']:<8} {r['trades_per_year']:>9.1f}회 {avg_return:>+11.2f}% "
              f"{r['compounded_annual']:>+11.1f}% {compounded_10y:>+13.1f}%")
    
    # 3. 물타기 강도
    print("\n### 3️⃣ 물타기 강도")
    print("-" * 80)
    print(f"{'종목':<8} {'평균물타기':>10} {'최대물타기':>10} {'평균보유':>12} {'최대보유':>12}")
    print("-" * 80)
    for r in results:
        print(f"{r['ticker']:<8} {r['avg_buys']:>9.1f}회 {r['max_buys']:>9}회 "
              f"{r['avg_holding']:>10.0f}일 {r['max_holding']:>10}일")
    
    # 4. 리스크
    print("\n### 4️⃣ 리스크 지표")
    print("-" * 80)
    print(f"{'종목':<8} {'최대손실':>10} {'손절횟수':>8} {'손절손실':>12} {'샤프비율':>10}")
    print("-" * 80)
    for r in results:
        print(f"{r['ticker']:<8} {r['max_drawdown']:>+9.1f}% {r['stoploss_count']:>7}회 "
              f"${r['stoploss_loss']:>+10,.0f} {r['sharpe']:>9.2f}")
    
    # 5. 현재 상태
    print("\n### 5️⃣ 현재 보유 상태")
    print("-" * 80)
    for r in results:
        pos = r['current_position']
        if pos:
            print(f"{r['ticker']:<8}: {pos['num_buys']}회 물타기, ${pos['invested']:,} 투자, "
                  f"{pos['unrealized_return']:+.1f}% ({pos['holding_days']}일)")
        else:
            print(f"{r['ticker']:<8}: 대기 중")
    
    # 6. 종합 점수 (거래수 반영!)
    print("\n" + "="*80)
    print("🏆 종합 점수 (100점 만점, 거래수 반영)")
    print("="*80)
    
    scores = []
    for r in results:
        # 수익률 점수 (25점)
        return_score = min(25, max(0, r['total_return'] * 0.8))
        
        # 승률 점수 (15점)
        winrate_score = r['win_rate'] * 0.15
        
        # 거래수 점수 (20점) - 연간 거래수 기준 ⭐
        trade_score = min(20, r['trades_per_year'] * 10)  # 연 2회 = 20점
        
        # 물타기 효율 점수 (15점)
        water_score = max(0, 15 - r['avg_buys'] * 2)
        
        # 리스크 점수 (15점)
        dd_penalty = abs(r['max_drawdown']) * 0.4
        sl_penalty = r['stoploss_count'] * 2
        risk_score = max(0, 15 - dd_penalty - sl_penalty)
        
        # 효율성 점수 (10점)
        efficiency_score = min(10, r['sharpe'] * 3)
        
        total = return_score + winrate_score + trade_score + water_score + risk_score + efficiency_score
        
        scores.append({
            'ticker': r['ticker'],
            'return': return_score,
            'winrate': winrate_score,
            'trades': trade_score,
            'water': water_score,
            'risk': risk_score,
            'efficiency': efficiency_score,
            'total': total
        })
    
    scores.sort(key=lambda x: x['total'], reverse=True)
    
    print(f"\n{'종목':<8} {'수익률':>8} {'승률':>6} {'거래수':>8} {'물타기':>8} {'리스크':>8} {'효율':>6} {'총점':>8}")
    print("-" * 80)
    for s in scores:
        bar = '█' * int(s['total'] / 5)
        print(f"{s['ticker']:<8} {s['return']:>7.1f} {s['winrate']:>5.1f} {s['trades']:>7.1f} "
              f"{s['water']:>7.1f} {s['risk']:>7.1f} {s['efficiency']:>5.1f} {s['total']:>7.1f} {bar}")
    
    # 7. 거래 내역 (최근 5건씩)
    print("\n" + "="*80)
    print("📋 최근 거래 내역 (각 5건)")
    print("="*80)
    
    for r in results:
        print(f"\n【{r['ticker']}】")
        print("-" * 60)
        recent = r['trades'][-5:] if len(r['trades']) >= 5 else r['trades']
        for t in reversed(recent):
            start = t['entry_dates'][0].strftime('%Y-%m-%d')
            end = t['exit_date'].strftime('%Y-%m-%d')
            print(f"  {start} ~ {end}: {t['num_buys']}회 물타기, "
                  f"${t['invested']:,} → ${t['profit']:+,.0f} ({t['return']:+.1f}%)")


def compare_with_existing():
    """기존 종목들과 비교"""
    print("\n" + "="*80)
    print("📊 기존 종목 vs 대안 종목 비교")
    print("="*80)
    
    # 기존 종목 결과 (이전 분석에서 가져옴)
    existing = {
        'AAPL': {'trades': 10, 'per_year': 1.1, 'return': 28.1, 'avg_buys': 1.9, 'max_buys': 3},
        'QQQ': {'trades': 10, 'per_year': 1.2, 'return': 20.8, 'avg_buys': 4.0, 'max_buys': 10},
        'SMH': {'trades': 10, 'per_year': 1.1, 'return': 33.5, 'avg_buys': 2.9, 'max_buys': 8},
    }
    
    print("\n기존 종목 (참고):")
    print("-" * 60)
    for ticker, data in existing.items():
        print(f"  {ticker}: {data['trades']}회 거래, 연 {data['per_year']:.1f}회, "
              f"+{data['return']:.1f}%, 평균 {data['avg_buys']:.1f}회 물타기")


def main():
    print("="*80)
    print("🔍 대안 종목 백테스트")
    print("="*80)
    print(f"종목: {', '.join(TICKERS)}")
    print(f"전략: {STRATEGY['name']} (RSI {STRATEGY['rsi_oversold']}/{STRATEGY['rsi_buy_exit']} → "
          f"{STRATEGY['rsi_overbought']}/{STRATEGY['rsi_sell_exit']})")
    print(f"기간: 10년")
    
    results = []
    
    for ticker in TICKERS:
        df = load_data(ticker, '10y')
        
        if df is None:
            print(f"  ❌ {ticker} 데이터 로드 실패")
            continue
        
        # AAPL 전략으로 백테스트
        buy_signals = find_buy_signals(df, STRATEGY)
        sell_signals = find_sell_signals(df, STRATEGY)
        trades, current_pos, max_dd = simulate_trades(df, buy_signals, sell_signals, STRATEGY)
        
        result = analyze_results(ticker, df, trades, current_pos, max_dd, STRATEGY)
        if result:
            results.append(result)
    
    if results:
        print_results(results)
        compare_with_existing()
    
    # QQQ 전략으로도 비교
    print("\n" + "="*80)
    print("📊 QQQ 전략으로 비교 (RSI 35/40 → 75/50)")
    print("="*80)
    
    results_qqq = []
    for ticker in TICKERS:
        df = load_data(ticker, '10y')
        if df is None:
            continue
        
        buy_signals = find_buy_signals(df, QQQ_STRATEGY)
        sell_signals = find_sell_signals(df, QQQ_STRATEGY)
        trades, current_pos, max_dd = simulate_trades(df, buy_signals, sell_signals, QQQ_STRATEGY)
        
        result = analyze_results(ticker, df, trades, current_pos, max_dd, QQQ_STRATEGY)
        if result:
            results_qqq.append(result)
    
    if results_qqq:
        print("\n### QQQ 전략 결과")
        print("-" * 80)
        print(f"{'종목':<8} {'거래수':>8} {'연간거래':>10} {'승률':>8} {'총수익률':>10} {'평균물타기':>10}")
        print("-" * 80)
        for r in results_qqq:
            print(f"{r['ticker']:<8} {r['total_trades']:>7}회 {r['trades_per_year']:>9.1f}회 "
                  f"{r['win_rate']:>7.0f}% {r['total_return']:>+9.1f}% {r['avg_buys']:>9.1f}회")
    
    # 최종 추천
    print("\n" + "="*80)
    print("💡 최종 추천")
    print("="*80)
    
    if results:
        best = max(results, key=lambda x: x['total_return'])
        most_trades = max(results, key=lambda x: x['trades_per_year'])
        lowest_water = min(results, key=lambda x: x['avg_buys'])
        
        print(f"\n🏆 최고 수익률: {best['ticker']} (+{best['total_return']:.1f}%)")
        print(f"📈 최다 거래: {most_trades['ticker']} (연 {most_trades['trades_per_year']:.1f}회)")
        print(f"💧 최소 물타기: {lowest_water['ticker']} (평균 {lowest_water['avg_buys']:.1f}회)")


if __name__ == "__main__":
    main()

