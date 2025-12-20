#!/usr/bin/env python3
"""
Auto-Stock: 검증된 패턴 기반 주식 매매 신호 시스템

검증된 14개 패턴:
- Combo_Strong_Dip: Test 승률 100%, Lift 2.84x
- Momentum20_Negative: Test 승률 92.3%, Lift 2.62x
- RSI_Oversold_35: Test 승률 73.9%, Lift 2.10x
- ... 등

사용법:
    python main.py                    # 신호 확인 (QQQ)
    python main.py --ticker SPY       # 특정 종목
    python main.py --discover         # 패턴 재발견 (처음 또는 재검증 시)
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.cache import DataCache
from src.data.fetcher import DataFetcher
from src.data.validator import DataValidator
from src.features.technical import TechnicalIndicators
from src.discovery.validated_patterns import get_validated_patterns, check_signals, VALIDATED_PATTERNS
from src.utils.helpers import load_config


def run_signal_check(ticker: str, config: dict, lookback_days: int = 7):
    """신호 확인"""
    print("\n" + "="*70)
    print(f"🚀 Auto-Stock 신호 확인: {ticker}")
    print(f"   검증된 패턴: {len(VALIDATED_PATTERNS)}개")
    print(f"   분석 기간: 최근 {lookback_days}일")
    print("="*70)
    
    # 데이터 로드
    print("\n[1/3] 📥 데이터 로드...")
    cache = DataCache(
        cache_dir=str(project_root / config['data']['cache']['directory']),
        max_age_hours=config['data']['cache']['max_age_hours']
    )
    
    df = cache.get(ticker)
    if df is None:
        print(f"   캐시 없음, 새로 다운로드...")
        fetcher = DataFetcher([ticker])
        data = fetcher.fetch(config['data']['period'])
        if ticker not in data:
            print(f"❌ {ticker} 데이터 없음")
            return
        df = data[ticker]
        df, _ = DataValidator.validate(df, ticker)
        cache.set(ticker, df)
    
    print(f"   ✅ {len(df)} 거래일 로드 ({df.index[0].date()} ~ {df.index[-1].date()})")
    
    # 지표 계산
    print("\n[2/3] 📊 지표 계산...")
    indicators = TechnicalIndicators(config.get('indicators', {}))
    df = indicators.calculate_all(df)
    print(f"   ✅ 기술적 지표 계산 완료")
    
    # 현재 상태
    current = df['Close'].iloc[-1]
    prev = df['Close'].iloc[-2]
    change = (current / prev - 1) * 100
    rsi = df['rsi'].iloc[-1]
    momentum = df['momentum_10'].iloc[-1]
    
    print(f"\n   현재가: ${current:.2f} ({change:+.2f}%)")
    print(f"   RSI: {rsi:.1f}, 모멘텀(10일): {momentum:.1f}%")
    
    # 신호 확인
    print(f"\n[3/3] 📡 신호 확인 (최근 {lookback_days}일)...")
    signals = check_signals(df, lookback_days)
    
    if signals:
        today_signals = [s for s in signals if s['days_ago'] == 0]
        past_signals = [s for s in signals if s['days_ago'] > 0]
        
        print(f"\n   📊 총 {len(signals)}개 신호 발생")
        
        if today_signals:
            print(f"\n   🟢 오늘 신호 ({len(today_signals)}개):")
            for s in today_signals:
                print(f"      - {s['pattern']}: 승률 {s['test_win_rate']*100:.0f}%, 평균 {s['test_avg_return']:.1f}%")
        
        if past_signals:
            print(f"\n   📌 최근 신호:")
            for s in past_signals[:5]:  # 최근 5개만
                print(f"      - D-{s['days_ago']} ({s['date'].strftime('%m/%d')}): {s['pattern']} (승률 {s['test_win_rate']*100:.0f}%)")
    else:
        print(f"\n   📭 최근 {lookback_days}일간 신호 없음")
    
    # 결과 저장
    result = {
        'ticker': ticker,
        'timestamp': datetime.now().isoformat(),
        'current_price': current,
        'change_pct': change,
        'rsi': rsi,
        'momentum_10': momentum,
        'signals': [
            {
                'pattern': s['pattern'],
                'date': s['date'].isoformat(),
                'days_ago': s['days_ago'],
                'price': s['price'],
                'test_win_rate': s['test_win_rate'],
                'test_avg_return': s['test_avg_return']
            }
            for s in signals
        ]
    }
    
    results_dir = project_root / "data" / "signals"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / f"{ticker}_signals.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    # 리포트
    print("\n" + "="*70)
    print(f"📊 {ticker} 리포트")
    print("="*70)
    
    if today_signals:
        print(f"\n⚡ 오늘 {len(today_signals)}개 매수 신호 발생!")
        best = max(today_signals, key=lambda x: x['test_win_rate'])
        print(f"   최고 신호: {best['pattern']} (승률 {best['test_win_rate']*100:.0f}%)")
    else:
        print(f"\n📭 오늘은 매수 신호 없음")
    
    print("="*70)
    
    return result


def run_pattern_discovery(ticker: str, config: dict):
    """패턴 발견 실행"""
    from src.discovery.pattern_finder import run_full_pipeline
    
    print("\n" + "="*70)
    print(f"🔬 패턴 발견 시작: {ticker}")
    print("="*70)
    
    # 데이터 로드
    cache = DataCache(
        cache_dir=str(project_root / config['data']['cache']['directory']),
        max_age_hours=config['data']['cache']['max_age_hours']
    )
    
    df = cache.get(ticker)
    if df is None:
        fetcher = DataFetcher([ticker])
        data = fetcher.fetch(config['data']['period'])
        df = data[ticker]
        df, _ = DataValidator.validate(df, ticker)
        cache.set(ticker, df)
    
    # 지표 계산
    indicators = TechnicalIndicators(config.get('indicators', {}))
    df = indicators.calculate_all(df)
    
    # 패턴 발견 + 검증
    patterns, info = run_full_pipeline(df, holding_period=60, min_return=10.0)
    
    # 결과 저장
    results_dir = project_root / "data" / "patterns"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / f"{ticker}_discovery.json", 'w') as f:
        json.dump(info, f, indent=2, default=str)
    
    print(f"\n💾 결과 저장: {results_dir / f'{ticker}_discovery.json'}")
    
    return patterns, info


def main():
    parser = argparse.ArgumentParser(
        description='Auto-Stock: 검증된 패턴 기반 매매 신호 시스템',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
검증된 패턴 (상위 5개):
  - Combo_Strong_Dip: Test 승률 100%, Lift 2.84x
  - Momentum20_Negative: Test 승률 92.3%, Lift 2.62x
  - RSI_Oversold_35: Test 승률 73.9%, Lift 2.10x
  - BB_BelowLower: Test 승률 73.9%, Lift 2.10x
  - Price_Below_MA20_5pct: Test 승률 73.9%, Lift 2.10x

예시:
    python main.py                    # QQQ 신호 확인
    python main.py --ticker SPY       # SPY 신호 확인
    python main.py --discover         # 패턴 재발견
        """
    )
    
    parser.add_argument(
        '--ticker', '-t',
        type=str,
        default=None,
        help='분석할 종목 (기본: QQQ)'
    )
    
    parser.add_argument(
        '--discover', '-d',
        action='store_true',
        help='패턴 발견 모드 (처음 또는 재검증 시)'
    )
    
    parser.add_argument(
        '--lookback', '-l',
        type=int,
        default=7,
        help='신호 확인 기간 (기본: 7일)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='설정 파일 경로'
    )
    
    args = parser.parse_args()
    
    try:
        # 설정 로드
        config = load_config(args.config)
        ticker = args.ticker or config['tickers'][0]
        
        if args.discover:
            # 패턴 발견 모드
            run_pattern_discovery(ticker, config)
        else:
            # 신호 확인 모드
            run_signal_check(ticker, config, args.lookback)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ 중단됨")
        return 130
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
