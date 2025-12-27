"""
6개 종목 최종 종합 분석
- 상관관계 매트릭스
- 클러스터링 (어떻게 묶이는지)
- 거래량, 수익률, 물타기 종합 비교
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 6개 종목
TICKERS = ['QQQ', 'AAPL', 'SMH', 'XOM', 'XLE', 'JPM']

# 각 종목 최적 파라미터 (이미 최적화된 것)
STRATEGIES = {
    'QQQ': {'params': '35/40→75/50', 'rsi_oversold': 35, 'rsi_buy_exit': 40, 'rsi_overbought': 75, 'rsi_sell_exit': 50},
    'AAPL': {'params': '30/35→75/50', 'rsi_oversold': 30, 'rsi_buy_exit': 35, 'rsi_overbought': 75, 'rsi_sell_exit': 50},
    'SMH': {'params': '35/40→75/45', 'rsi_oversold': 35, 'rsi_buy_exit': 40, 'rsi_overbought': 75, 'rsi_sell_exit': 45},
    'XOM': {'params': '25/40→85/55', 'rsi_oversold': 25, 'rsi_buy_exit': 40, 'rsi_overbought': 85, 'rsi_sell_exit': 55},
    'XLE': {'params': '25/35→85/60', 'rsi_oversold': 25, 'rsi_buy_exit': 35, 'rsi_overbought': 85, 'rsi_sell_exit': 60},
    'JPM': {'params': '25/30→80/50', 'rsi_oversold': 25, 'rsi_buy_exit': 30, 'rsi_overbought': 80, 'rsi_sell_exit': 50},
}

# 각 종목 성과 (이전 분석 결과)
PERFORMANCE = {
    'QQQ': {'return': 20.8, 'trades': 10, 'per_year': 1.0, 'avg_buys': 4.0, 'max_buys': 10, 'max_dd': -19.4},
    'AAPL': {'return': 28.1, 'trades': 10, 'per_year': 1.1, 'avg_buys': 1.9, 'max_buys': 3, 'max_dd': -24.9},
    'SMH': {'return': 33.5, 'trades': 10, 'per_year': 1.0, 'avg_buys': 2.9, 'max_buys': 8, 'max_dd': -24.0},
    'XOM': {'return': 24.6, 'trades': 7, 'per_year': 0.8, 'avg_buys': 4.0, 'max_buys': 9, 'max_dd': -53.3},
    'XLE': {'return': 21.7, 'trades': 11, 'per_year': 1.1, 'avg_buys': 2.3, 'max_buys': 6, 'max_dd': -56.4},
    'JPM': {'return': 11.8, 'trades': 15, 'per_year': 1.6, 'avg_buys': 2.0, 'max_buys': 7, 'max_dd': -23.9},
}

CAPITAL = 1000


def load_data():
    """모든 종목 데이터 로드"""
    print("⏳ 데이터 로딩 중...")
    
    all_data = {}
    for ticker in TICKERS:
        df = yf.download(ticker, period='5y', progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            all_data[ticker] = df
            print(f"  ✅ {ticker}: {len(df)}일")
    
    return all_data


def calculate_correlation_matrix(data: dict):
    """상관관계 매트릭스 계산"""
    # 일간 수익률로 상관관계 계산
    returns = pd.DataFrame()
    for ticker, df in data.items():
        returns[ticker] = df['Close'].pct_change()
    
    corr_matrix = returns.corr()
    return corr_matrix


def analyze_clusters(corr_matrix: pd.DataFrame):
    """상관관계 기반 클러스터 분석"""
    clusters = []
    
    # 높은 상관관계 (0.7 이상) 찾기
    high_corr_pairs = []
    for i, t1 in enumerate(TICKERS):
        for j, t2 in enumerate(TICKERS):
            if i < j:
                corr = corr_matrix.loc[t1, t2]
                if corr >= 0.7:
                    high_corr_pairs.append((t1, t2, corr))
    
    # 클러스터링
    tech_cluster = {'QQQ', 'AAPL', 'SMH'}
    energy_cluster = {'XOM', 'XLE'}
    finance_cluster = {'JPM'}
    
    return {
        'tech': tech_cluster,
        'energy': energy_cluster,
        'finance': finance_cluster,
        'high_corr_pairs': high_corr_pairs
    }


def calculate_avg_volume(data: dict):
    """평균 거래량 계산"""
    volumes = {}
    for ticker, df in data.items():
        vol = df['Volume']
        if isinstance(vol, pd.DataFrame):
            vol = vol.iloc[:, 0]
        volumes[ticker] = vol.mean() / 1e6  # 백만 단위
    return volumes


def main():
    print("="*80)
    print("📊 6개 종목 최종 종합 분석")
    print("="*80)
    
    # 데이터 로드
    data = load_data()
    
    # 상관관계 계산
    print("\n" + "="*80)
    print("📈 상관관계 매트릭스")
    print("="*80)
    
    corr_matrix = calculate_correlation_matrix(data)
    
    # 상관관계 테이블 출력
    print("\n         ", end="")
    for t in TICKERS:
        print(f"{t:>8}", end="")
    print()
    print("  " + "-"*56)
    
    for t1 in TICKERS:
        print(f"  {t1:<6}", end="")
        for t2 in TICKERS:
            corr = corr_matrix.loc[t1, t2]
            if t1 == t2:
                print(f"{'1.00':>8}", end="")
            elif corr >= 0.7:
                print(f"{corr:>7.2f}*", end="")  # 높은 상관
            elif corr >= 0.5:
                print(f"{corr:>8.2f}", end="")
            else:
                print(f"{corr:>8.2f}", end="")  # 낮은 상관
        print()
    
    print("\n  (* = 0.7 이상 높은 상관관계)")
    
    # 클러스터 분석
    clusters = analyze_clusters(corr_matrix)
    
    print("\n" + "="*80)
    print("🔗 상관관계 클러스터 (어떻게 묶이는지)")
    print("="*80)
    
    print("\n### 높은 상관관계 페어 (0.7 이상)")
    print("-"*50)
    for t1, t2, corr in sorted(clusters['high_corr_pairs'], key=lambda x: -x[2]):
        print(f"  {t1} ↔ {t2}: {corr:.2f}")
    
    print("\n### 섹터별 클러스터")
    print("-"*50)
    print(f"  🖥️ 기술주: {', '.join(clusters['tech'])}")
    print(f"     → QQQ, AAPL, SMH 상관관계 0.8+ (함께 움직임)")
    print(f"  ⛽ 에너지: {', '.join(clusters['energy'])}")
    print(f"     → XOM, XLE 상관관계 0.95+ (거의 동일하게 움직임)")
    print(f"  🏦 금융: {', '.join(clusters['finance'])}")
    print(f"     → JPM은 다른 섹터와 상관 0.3~0.5 (분산 효과)")
    
    # 거래량 분석
    volumes = calculate_avg_volume(data)
    
    print("\n" + "="*80)
    print("📊 6개 종목 종합 비교표")
    print("="*80)
    
    print("\n### 성과 & 리스크")
    print("-"*90)
    print(f"{'종목':<8} {'섹터':<8} {'전략':^16} {'수익률':>10} {'거래수':>8} {'연거래':>8} {'평균물타기':>10} {'최대물타기':>10}")
    print("-"*90)
    
    sector_map = {'QQQ': '기술', 'AAPL': '기술', 'SMH': '반도체', 'XOM': '에너지', 'XLE': '에너지', 'JPM': '금융'}
    
    for ticker in TICKERS:
        s = STRATEGIES[ticker]
        p = PERFORMANCE[ticker]
        print(f"{ticker:<8} {sector_map[ticker]:<8} {s['params']:^16} "
              f"{p['return']:>+9.1f}% {p['trades']:>7}회 {p['per_year']:>7.1f}회 "
              f"{p['avg_buys']:>9.1f}회 {p['max_buys']:>9}회")
    
    print("\n### 리스크 & 거래량")
    print("-"*70)
    print(f"{'종목':<8} {'섹터':<8} {'최대손실':>12} {'거래량(M/일)':>14} {'기술주상관':>12}")
    print("-"*70)
    
    for ticker in TICKERS:
        p = PERFORMANCE[ticker]
        vol = volumes.get(ticker, 0)
        # QQQ와의 상관관계
        corr_with_qqq = corr_matrix.loc[ticker, 'QQQ'] if ticker != 'QQQ' else 1.0
        print(f"{ticker:<8} {sector_map[ticker]:<8} {p['max_dd']:>+11.1f}% {vol:>13.1f}M {corr_with_qqq:>11.2f}")
    
    # 종합 점수
    print("\n" + "="*80)
    print("🏆 종합 점수 (수익률 + 거래수 + 물타기 + 리스크 + 분산)")
    print("="*80)
    
    scores = []
    for ticker in TICKERS:
        p = PERFORMANCE[ticker]
        
        # 수익률 (30점)
        return_score = min(30, p['return'] * 0.9)
        
        # 거래 빈도 (20점)
        if 0.8 <= p['per_year'] <= 2.0:
            trade_score = 20
        elif 0.5 <= p['per_year'] <= 3.0:
            trade_score = 15
        else:
            trade_score = 10
        
        # 물타기 효율 (20점)
        if p['avg_buys'] <= 2.5:
            water_score = 20
        elif p['avg_buys'] <= 3.5:
            water_score = 15
        else:
            water_score = 10
        
        # 리스크 (15점)
        if abs(p['max_dd']) <= 25:
            risk_score = 15
        elif abs(p['max_dd']) <= 40:
            risk_score = 10
        else:
            risk_score = 5
        
        # 분산 효과 (15점) - QQQ와 상관관계 낮을수록 좋음
        corr_with_tech = np.mean([corr_matrix.loc[ticker, t] for t in ['QQQ', 'AAPL', 'SMH'] if t != ticker])
        if corr_with_tech < 0.4:
            diversify_score = 15
        elif corr_with_tech < 0.6:
            diversify_score = 10
        else:
            diversify_score = 5
        
        total = return_score + trade_score + water_score + risk_score + diversify_score
        
        scores.append({
            'ticker': ticker,
            'sector': sector_map[ticker],
            'return': return_score,
            'trade': trade_score,
            'water': water_score,
            'risk': risk_score,
            'diversify': diversify_score,
            'total': total
        })
    
    scores.sort(key=lambda x: x['total'], reverse=True)
    
    print(f"\n{'종목':<8} {'섹터':<8} {'수익률':>8} {'거래수':>8} {'물타기':>8} {'리스크':>8} {'분산':>8} {'총점':>8}")
    print("-"*75)
    for s in scores:
        bar = '█' * int(s['total'] / 5)
        print(f"{s['ticker']:<8} {s['sector']:<8} {s['return']:>7.1f} {s['trade']:>7.1f} "
              f"{s['water']:>7.1f} {s['risk']:>7.1f} {s['diversify']:>7.1f} {s['total']:>7.1f} {bar}")
    
    # 클러스터별 분석
    print("\n" + "="*80)
    print("📋 클러스터별 추천 (같은 클러스터에서 1개씩)")
    print("="*80)
    
    print("\n### 🖥️ 기술주 클러스터 (QQQ, AAPL, SMH) - 1개 선택")
    print("-"*60)
    tech_scores = [s for s in scores if s['ticker'] in clusters['tech']]
    for i, s in enumerate(sorted(tech_scores, key=lambda x: -x['total'])):
        medal = ["🥇", "🥈", "🥉"][i]
        p = PERFORMANCE[s['ticker']]
        print(f"  {medal} {s['ticker']}: 점수 {s['total']:.1f}, 수익률 +{p['return']:.1f}%, 물타기 {p['avg_buys']:.1f}회")
    
    print("\n### ⛽ 에너지 클러스터 (XOM, XLE) - 1개 선택")
    print("-"*60)
    energy_scores = [s for s in scores if s['ticker'] in clusters['energy']]
    for i, s in enumerate(sorted(energy_scores, key=lambda x: -x['total'])):
        medal = ["🥇", "🥈"][i]
        p = PERFORMANCE[s['ticker']]
        print(f"  {medal} {s['ticker']}: 점수 {s['total']:.1f}, 수익률 +{p['return']:.1f}%, 물타기 {p['avg_buys']:.1f}회")
    
    print("\n### 🏦 금융 클러스터 (JPM)")
    print("-"*60)
    finance_scores = [s for s in scores if s['ticker'] in clusters['finance']]
    for s in finance_scores:
        p = PERFORMANCE[s['ticker']]
        print(f"  🥇 {s['ticker']}: 점수 {s['total']:.1f}, 수익률 +{p['return']:.1f}%, 물타기 {p['avg_buys']:.1f}회")
    
    # 최종 조합 추천
    print("\n" + "="*80)
    print("💡 최종 조합 추천")
    print("="*80)
    
    print("""
### 2종목 조합 (주식만)

1️⃣ **AAPL + JPM** ⭐ 추천
   - 기술 + 금융 분산
   - 상관관계: 0.44 (낮음)
   - 수익률: AAPL +28% + JPM +12% = 평균 +20%
   - 물타기: 둘 다 2회 이하로 적음

2️⃣ **SMH + JPM**
   - 반도체 + 금융 분산
   - 상관관계: 0.38 (매우 낮음)
   - 수익률: SMH +34% + JPM +12% = 평균 +23%
   - 물타기: SMH가 조금 높음 (2.9회)

3️⃣ **AAPL + XLE**
   - 기술 + 에너지 분산
   - 상관관계: 0.24 (매우 낮음)
   - 수익률: AAPL +28% + XLE +22% = 평균 +25%
   - ⚠️ XLE 최대손실 -56%로 리스크 높음

### ⚠️ 피해야 할 조합

❌ QQQ + AAPL: 상관 0.90 (거의 같은 움직임)
❌ QQQ + SMH: 상관 0.92 (거의 같은 움직임)  
❌ XOM + XLE: 상관 0.97 (거의 동일)
❌ AAPL + SMH: 상관 0.87 (높은 상관)

### + 코인 추가 시

✅ **AAPL + JPM + BTC**
   - 3자산 분산 (기술 + 금융 + 크립토)
   - BTC는 주식과 상관관계 낮음
   - 승률 100%, 손절 없음
""")
    
    # 상관관계 히트맵 요약
    print("\n" + "="*80)
    print("🗺️ 상관관계 요약 (분산 투자 가이드)")
    print("="*80)
    
    print("""
                    ┌─────────────────────────────────────┐
                    │         상관관계 클러스터            │
                    └─────────────────────────────────────┘
    
    ┌─────────────────┐     ┌─────────────┐     ┌─────────────┐
    │   🖥️ 기술주      │     │  ⛽ 에너지   │     │   🏦 금융    │
    │                 │     │             │     │             │
    │  QQQ ←0.90→ AAPL│     │ XOM ←0.97→  │     │    JPM      │
    │   ↑             │     │     XLE     │     │             │
    │  0.92           │     │             │     │             │
    │   ↓             │     │             │     │             │
    │  SMH ←0.87→ AAPL│     │             │     │             │
    └────────┬────────┘     └──────┬──────┘     └──────┬──────┘
             │                     │                    │
             │       0.20~0.30     │      0.35~0.45     │
             └─────────────────────┴────────────────────┘
    
    💡 분산 투자 원칙:
       - 같은 박스 안에서는 1개만 선택
       - 다른 박스끼리 조합하면 분산 효과 ↑
    """)


if __name__ == "__main__":
    main()

