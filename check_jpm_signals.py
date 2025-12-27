"""
JPM 대시보드 로직과 동일하게 매수 시그널 확인
"""
import sys
sys.path.insert(0, '.')

from src.data.cache import DataCache
from src.data.fetcher import DataFetcher
from src.data.validator import DataValidator
from src.features.technical import TechnicalIndicators
from src.utils.helpers import load_config
import pandas as pd

# JPM 전략 파라미터 (대시보드와 동일)
TICKER = "JPM"
RSI_OVERSOLD = 25
RSI_BUY_EXIT = 30
RSI_OVERBOUGHT = 80
RSI_SELL_EXIT = 50


def main():
    config = load_config()
    
    # 데이터 로드 (대시보드와 동일한 방식)
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
    
    print(f"데이터 기간: {df.index[0].date()} ~ {df.index[-1].date()}")
    print(f"총 {len(df)}일")
    
    # RSI 통계
    print(f"\nRSI 통계:")
    print(f"  RSI < 25: {(df['rsi'] < 25).sum()}회 ({(df['rsi'] < 25).sum()/len(df)*100:.1f}%)")
    print(f"  RSI < 30: {(df['rsi'] < 30).sum()}회 ({(df['rsi'] < 30).sum()/len(df)*100:.1f}%)")
    print(f"  RSI < 35: {(df['rsi'] < 35).sum()}회 ({(df['rsi'] < 35).sum()/len(df)*100:.1f}%)")
    
    # 매수 시그널 (대시보드와 동일한 로직)
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
    
    print(f"\n총 매수 시그널: {len(buy_signals)}개")
    print("-"*80)
    print(f"{'#':<4} {'과매도진입일':^14} {'확정일':^14} {'확정가':>10} {'RSI':>8}")
    print("-"*80)
    
    for i, bs in enumerate(buy_signals):
        print(f"{i+1:<4} {bs['signal_date'].strftime('%Y-%m-%d'):^14} "
              f"{bs['confirm_date'].strftime('%Y-%m-%d'):^14} "
              f"${bs['confirm_price']:.2f}:>10 {bs['rsi_at_confirm']:.1f}:>8")
    
    # 연도별 분포
    print("\n연도별 매수 시그널:")
    from collections import Counter
    years = [bs['confirm_date'].year for bs in buy_signals]
    year_counts = Counter(years)
    for year in sorted(year_counts.keys()):
        print(f"  {year}: {year_counts[year]}회")


if __name__ == "__main__":
    main()

