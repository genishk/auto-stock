"""
GitHub Actions용 일일 시그널 체크 스크립트
"""
import sys
sys.path.insert(0, '.')

from src.data.cache import DataCache
from src.data.fetcher import DataFetcher
from src.data.validator import DataValidator
from src.features.technical import TechnicalIndicators
from src.discovery.validated_patterns import VALIDATED_PATTERNS
from src.utils.helpers import load_config
from datetime import datetime
import json

def main():
    config = load_config()
    ticker = 'QQQ'
    
    # 데이터 로드
    cache = DataCache(cache_dir='data/cache', max_age_hours=24)
    df = cache.get(ticker)
    if df is None:
        fetcher = DataFetcher([ticker])
        data = fetcher.fetch('10y')
        df = data[ticker]
        df, _ = DataValidator.validate(df, ticker)
        cache.set(ticker, df)
    
    # 기술 지표 계산
    ti = TechnicalIndicators(df)
    df = ti.add_all_indicators()
    
    # 최신 데이터
    latest = df.iloc[-1]
    current_date = df.index[-1].strftime('%Y-%m-%d')
    current_rsi = latest.get('RSI', 0)
    current_price = latest['Close']
    
    # 가격 정보
    open_price = latest['Open']
    high_price = latest['High']
    low_price = latest['Low']
    close_price = latest['Close']
    
    # 시그널 체크
    buy_signal = False
    sell_signal = False
    
    # 매수 시그널: RSI < 35 후 RSI >= 60으로 탈출
    rsi_oversold_threshold = 35
    rsi_buy_exit_threshold = 60
    
    # 매도 시그널: RSI > 70 후 RSI <= 50으로 하락
    rsi_overbought_threshold = 70
    rsi_sell_exit_threshold = 50
    
    # 최근 데이터에서 시그널 확인
    lookback = min(30, len(df))
    recent_df = df.iloc[-lookback:]
    
    # 매수 시그널 확인 (RSI 과매도 후 탈출)
    in_oversold = False
    for i in range(len(recent_df) - 1):
        rsi = recent_df['RSI'].iloc[i]
        if rsi < rsi_oversold_threshold:
            in_oversold = True
        elif in_oversold and rsi >= rsi_buy_exit_threshold:
            # 오늘이 탈출 시점인지 확인
            if i == len(recent_df) - 2:  # 어제 탈출
                buy_signal = True
            in_oversold = False
    
    # 오늘 탈출 확인
    if in_oversold and current_rsi >= rsi_buy_exit_threshold:
        buy_signal = True
    
    # 매도 시그널 확인 (RSI 과매수 후 하락)
    in_overbought = False
    for i in range(len(recent_df) - 1):
        rsi = recent_df['RSI'].iloc[i]
        if rsi > rsi_overbought_threshold:
            in_overbought = True
        elif in_overbought and rsi <= rsi_sell_exit_threshold:
            if i == len(recent_df) - 2:
                sell_signal = True
            in_overbought = False
    
    if in_overbought and current_rsi <= rsi_sell_exit_threshold:
        sell_signal = True
    
    # 결과 출력
    print('=' * 50)
    print('📊 Auto-Stock 일일 리포트')
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
    print(f'매수 기준: RSI < {rsi_oversold_threshold} → RSI >= {rsi_buy_exit_threshold}')
    print(f'매도 기준: RSI > {rsi_overbought_threshold} → RSI <= {rsi_sell_exit_threshold}')
    print()
    print('🚨 시그널')
    print('-' * 40)
    
    if buy_signal:
        print(f'🟢 매수 시그널 발생!')
        print(f'   RSI가 {rsi_oversold_threshold} 이하에서 {rsi_buy_exit_threshold} 이상으로 탈출')
        print(f'   현재 가격: ${current_price:.2f}')
    elif sell_signal:
        print(f'🔴 매도 시그널 발생!')
        print(f'   RSI가 {rsi_overbought_threshold} 이상에서 {rsi_sell_exit_threshold} 이하로 하락')
        print(f'   현재 가격: ${current_price:.2f}')
    else:
        print('📭 오늘은 시그널 없음')
    
    print()
    print('=' * 50)
    
    # GitHub Actions 출력 변수 설정
    if buy_signal:
        print('::set-output name=signal_type::buy')
        print(f'::set-output name=signal_price::{current_price:.2f}')
    elif sell_signal:
        print('::set-output name=signal_type::sell')
        print(f'::set-output name=signal_price::{current_price:.2f}')
    else:
        print('::set-output name=signal_type::none')
    
    # 환경 변수로도 설정 (새로운 방식)
    import os
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
            f.write(f'rsi_buy_threshold={rsi_oversold_threshold}\n')
            f.write(f'rsi_buy_exit={rsi_buy_exit_threshold}\n')
            f.write(f'rsi_sell_threshold={rsi_overbought_threshold}\n')
            f.write(f'rsi_sell_exit={rsi_sell_exit_threshold}\n')

if __name__ == '__main__':
    main()

