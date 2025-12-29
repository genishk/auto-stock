"""
GLD 일일 시그널 체크 스크립트
최적화 전략: RSI 40/50 → 65/60, GC OFF (거래 늘린 안전 버전)

시그널 vs 액션 구분:
- 시그널: RSI 기준으로 매수/매도 조건 충족
- 액션: 실제로 행동해야 하는지 (포지션 유무, 수익 여부 고려)
"""
import sys
sys.path.insert(0, '.')

from src.data.cache import DataCache
from src.data.fetcher import DataFetcher
from src.data.validator import DataValidator
from src.features.technical import TechnicalIndicators
from src.utils.helpers import load_config
from datetime import datetime
import pandas as pd
import os

# GLD 전략 파라미터
TICKER = "GLD"
RSI_OVERSOLD = 40
RSI_BUY_EXIT = 50
RSI_OVERBOUGHT = 65
RSI_SELL_EXIT = 60
USE_GOLDEN_CROSS = False
CAPITAL_PER_ENTRY = 1000


def find_buy_signals(df):
    """매수 시그널 찾기 (대시보드와 동일 로직)"""
    buy_signals = []
    in_oversold = False
    last_signal_date = None
    last_signal_price = None
    
    for idx in range(len(df)):
        rsi = df['rsi'].iloc[idx]
        if pd.isna(rsi):
            continue
        
        if rsi < RSI_OVERSOLD:
            in_oversold = True
            last_signal_date = df.index[idx]
            last_signal_price = df['Close'].iloc[idx]
        else:
            if in_oversold and rsi >= RSI_BUY_EXIT and last_signal_date is not None:
                buy_signals.append({
                    'signal_date': last_signal_date,
                    'signal_price': last_signal_price,
                    'confirm_date': df.index[idx],
                    'confirm_price': df['Close'].iloc[idx],
                    'rsi_at_confirm': rsi
                })
                in_oversold = False
                last_signal_date = None
    
    return buy_signals


def find_sell_signals(df):
    """매도 시그널 찾기 (대시보드와 동일 로직)"""
    sell_signals = []
    in_overbought = False
    last_signal_date = None
    last_signal_price = None
    
    for idx in range(len(df)):
        rsi = df['rsi'].iloc[idx]
        if pd.isna(rsi):
            continue
        
        if rsi > RSI_OVERBOUGHT:
            in_overbought = True
            last_signal_date = df.index[idx]
            last_signal_price = df['Close'].iloc[idx]
        else:
            if in_overbought and rsi <= RSI_SELL_EXIT and last_signal_date is not None:
                sell_signals.append({
                    'signal_date': last_signal_date,
                    'signal_price': last_signal_price,
                    'confirm_date': df.index[idx],
                    'confirm_price': df['Close'].iloc[idx]
                })
                in_overbought = False
                last_signal_date = None
    
    return sell_signals


def simulate_trades(df, buy_signals, sell_signals):
    """거래 시뮬레이션 (대시보드와 동일 로직) - 동일 금액, profit_only"""
    all_buy_dates = {bs['confirm_date']: bs for bs in buy_signals}
    all_sell_dates = {ss['confirm_date']: ss for ss in sell_signals}
    
    trades = []
    positions = []
    
    for idx in range(len(df)):
        current_date = df.index[idx]
        current_price = df['Close'].iloc[idx]
        
        if positions:
            n = len(positions)
            total_inv = n * CAPITAL_PER_ENTRY
            total_qty = sum(CAPITAL_PER_ENTRY / p['price'] for p in positions)
            avg_price = total_inv / total_qty
            
            if current_date in all_sell_dates:
                sell_price = all_sell_dates[current_date]['confirm_price']
                sell_return = (sell_price / avg_price - 1) * 100
                if sell_return > 0:  # profit_only
                    trades.append({
                        'entry_dates': [p['date'] for p in positions],
                        'entry_prices': [p['price'] for p in positions],
                        'avg_price': avg_price,
                        'num_buys': n,
                        'exit_date': current_date,
                        'exit_price': sell_price,
                        'return': sell_return,
                        'exit_reason': '익절'
                    })
                    positions = []
        
        if current_date in all_buy_dates:
            positions.append({
                'date': current_date,
                'price': all_buy_dates[current_date]['confirm_price']
            })
    
    return trades, positions


def main():
    config = load_config()
    
    # 데이터 로드
    cache = DataCache(cache_dir='data/cache', max_age_hours=24)
    df = cache.get(TICKER)
    if df is None:
        fetcher = DataFetcher([TICKER])
        data = fetcher.fetch('10y')
        df = data[TICKER]
        df, _ = DataValidator.validate(df, TICKER)
        cache.set(TICKER, df)
    
    # 기술 지표 계산
    ti = TechnicalIndicators(config.get('indicators', {}))
    df = ti.calculate_all(df)
    
    # 골든크로스용 이동평균선
    df['MA40'] = df['Close'].rolling(window=40).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['golden_cross'] = df['MA40'] > df['MA200']
    
    # 최신 데이터
    latest = df.iloc[-1]
    current_date = df.index[-1].strftime('%Y-%m-%d')
    current_rsi = latest.get('rsi', 0)
    current_price = latest['Close']
    
    open_price = latest['Open']
    high_price = latest['High']
    low_price = latest['Low']
    close_price = latest['Close']
    
    current_gc = latest.get('golden_cross', False)
    if pd.isna(current_gc):
        current_gc = False
    
    # 시그널 및 거래 시뮬레이션 (대시보드와 동일)
    buy_signals = find_buy_signals(df)
    sell_signals = find_sell_signals(df)
    trades, positions = simulate_trades(df, buy_signals, sell_signals)
    
    # 오늘 시그널 확인
    today = df.index[-1]
    buy_signal = any(bs['confirm_date'] == today for bs in buy_signals)
    sell_signal = any(ss['confirm_date'] == today for ss in sell_signals)
    
    # 액션 판단 (시그널과 별도)
    action = 'none'
    action_detail = ''
    
    # 포지션 상태 계산
    has_position = len(positions) > 0
    position_count = len(positions)
    avg_price = 0
    unrealized_pct = 0
    total_invested = 0
    
    if has_position:
        total_invested = position_count * CAPITAL_PER_ENTRY
        total_qty = sum(CAPITAL_PER_ENTRY / p['price'] for p in positions)
        avg_price = total_invested / total_qty
        unrealized_pct = (current_price / avg_price - 1) * 100
    
    if buy_signal:
        if has_position:
            action = 'add'  # 물타기
            action_detail = f'물타기 추가 ({position_count}→{position_count+1}회)'
        else:
            action = 'buy'  # 신규 매수
            action_detail = '신규 매수'
    elif sell_signal:
        if has_position:
            if unrealized_pct > 0:
                action = 'sell'  # 익절 매도
                action_detail = f'익절 매도 (수익률 {unrealized_pct:+.1f}%)'
            else:
                action = 'hold'  # 손실이라 홀드
                action_detail = f'매도 시그널이지만 손실 중 ({unrealized_pct:+.1f}%) → 홀드'
        else:
            action = 'skip'  # 포지션 없음
            action_detail = '매도 시그널이지만 보유 포지션 없음 → 무시'
    
    # 결과 출력
    print('=' * 50)
    print(f'🥇 {TICKER} 일일 리포트')
    print('=' * 50)
    print()
    print(f'📅 날짜: {current_date}')
    print()
    print('💰 가격 정보')
    print('-' * 40)
    print(f'시가: ${open_price:.2f}')
    print(f'고가: ${high_price:.2f}')
    print(f'저가: ${low_price:.2f}')
    print(f'종가: ${close_price:.2f}')
    print()
    print('📈 기술 지표')
    print('-' * 40)
    print(f'RSI: {current_rsi:.1f}')
    print()
    print('📊 현재 포지션')
    print('-' * 40)
    if has_position:
        print(f'보유 상태: {position_count}회 물타기 (${total_invested:,} 투자)')
        print(f'평균 매수가: ${avg_price:.2f}')
        print(f'미실현 손익: {unrealized_pct:+.1f}%')
    else:
        print('보유 상태: 대기 중 (포지션 없음)')
    print()
    print('🚨 시그널 & 액션')
    print('-' * 40)
    
    if buy_signal:
        print(f'📡 시그널: 🟢 매수 시그널 발생')
        print(f'🎯 액션: {action_detail}')
    elif sell_signal:
        print(f'📡 시그널: 🔴 매도 시그널 발생')
        print(f'🎯 액션: {action_detail}')
    else:
        print('📡 시그널: 없음')
        if has_position:
            print(f'🎯 액션: 홀드 (보유 중)')
        else:
            print('🎯 액션: 대기')
    
    print()
    print('=' * 50)
    
    # GitHub Actions 환경 변수
    github_output = os.environ.get('GITHUB_OUTPUT', '')
    if github_output:
        with open(github_output, 'a') as f:
            # 시그널 정보
            if buy_signal:
                f.write('signal_type=buy\n')
            elif sell_signal:
                f.write('signal_type=sell\n')
            else:
                f.write('signal_type=none\n')
            
            # 액션 정보
            f.write(f'action={action}\n')
            f.write(f'action_detail={action_detail}\n')
            
            # 포지션 정보
            f.write(f'has_position={"yes" if has_position else "no"}\n')
            f.write(f'position_count={position_count}\n')
            f.write(f'avg_price={avg_price:.2f}\n')
            f.write(f'unrealized_pct={unrealized_pct:.1f}\n')
            f.write(f'total_invested={total_invested}\n')
            
            # 기본 정보
            f.write(f'signal_price={current_price:.2f}\n')
            f.write(f'current_date={current_date}\n')
            f.write(f'current_rsi={current_rsi:.1f}\n')
            f.write(f'open_price={open_price:.2f}\n')
            f.write(f'high_price={high_price:.2f}\n')
            f.write(f'low_price={low_price:.2f}\n')
            f.write(f'close_price={close_price:.2f}\n')
            f.write(f'ticker={TICKER}\n')


if __name__ == '__main__':
    main()
