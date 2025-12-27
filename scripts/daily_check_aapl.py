"""
AAPL 일일 시그널 체크 스크립트
최적화 전략: RSI 30/35 → 75/50, GC OFF
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

# AAPL 전략 파라미터
TICKER = "AAPL"
RSI_OVERSOLD = 30
RSI_BUY_EXIT = 35
RSI_OVERBOUGHT = 75
RSI_SELL_EXIT = 50
USE_GOLDEN_CROSS = False


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
    ma40 = latest.get('MA40', 0)
    ma200 = latest.get('MA200', 0)
    
    # 시그널 체크
    buy_signal = False
    sell_signal = False
    
    lookback = min(30, len(df))
    recent_df = df.iloc[-lookback:]
    
    # 매수 시그널 확인
    in_oversold = False
    for i in range(len(recent_df) - 1):
        rsi = recent_df['rsi'].iloc[i]
        if rsi < RSI_OVERSOLD:
            in_oversold = True
        elif in_oversold and rsi >= RSI_BUY_EXIT:
            if i == len(recent_df) - 2:
                buy_signal = True
            in_oversold = False
    
    if in_oversold and current_rsi >= RSI_BUY_EXIT:
        buy_signal = True
    
    # 매도 시그널 확인
    in_overbought = False
    for i in range(len(recent_df) - 1):
        rsi = recent_df['rsi'].iloc[i]
        if rsi > RSI_OVERBOUGHT:
            in_overbought = True
        elif in_overbought and rsi <= RSI_SELL_EXIT:
            if i == len(recent_df) - 2:
                sell_signal = True
            in_overbought = False
    
    if in_overbought and current_rsi <= RSI_SELL_EXIT:
        sell_signal = True
    
    # 결과 출력
    print('=' * 50)
    print(f'🍎 {TICKER} 일일 리포트')
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
    print(f'MA40: ${ma40:.2f}' if not pd.isna(ma40) else 'MA40: N/A')
    print(f'MA200: ${ma200:.2f}' if not pd.isna(ma200) else 'MA200: N/A')
    print(f'골든크로스: {"🟢 상승장" if current_gc else "🔴 하락장"}')
    print()
    print(f'매수 기준: RSI < {RSI_OVERSOLD} → RSI >= {RSI_BUY_EXIT}')
    print(f'매도 기준: RSI > {RSI_OVERBOUGHT} → RSI <= {RSI_SELL_EXIT}')
    print(f'손절: 없음 (10년 승률 100%)')
    print()
    print('🚨 시그널')
    print('-' * 40)
    
    if buy_signal:
        print(f'🟢 매수 시그널 발생!')
        print(f'   RSI가 {RSI_OVERSOLD} 이하에서 {RSI_BUY_EXIT} 이상으로 탈출')
        print(f'   현재 가격: ${current_price:.2f}')
    elif sell_signal:
        print(f'🔴 매도 시그널 발생!')
        print(f'   RSI가 {RSI_OVERBOUGHT} 이상에서 {RSI_SELL_EXIT} 이하로 하락')
        print(f'   현재 가격: ${current_price:.2f}')
    else:
        print('📭 오늘은 시그널 없음')
    
    print()
    print('=' * 50)
    
    # GitHub Actions 환경 변수
    github_output = os.environ.get('GITHUB_OUTPUT', '')
    if github_output:
        with open(github_output, 'a') as f:
            if buy_signal:
                f.write(f'signal_type=buy\n')
                f.write(f'signal_price={current_price:.2f}\n')
            elif sell_signal:
                f.write(f'signal_type=sell\n')
                f.write(f'signal_price={current_price:.2f}\n')
            else:
                f.write('signal_type=none\n')
            f.write(f'current_date={current_date}\n')
            f.write(f'current_rsi={current_rsi:.1f}\n')
            f.write(f'open_price={open_price:.2f}\n')
            f.write(f'high_price={high_price:.2f}\n')
            f.write(f'low_price={low_price:.2f}\n')
            f.write(f'close_price={close_price:.2f}\n')
            f.write(f'ticker={TICKER}\n')


if __name__ == '__main__':
    main()

