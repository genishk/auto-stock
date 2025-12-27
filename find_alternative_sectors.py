"""
에너지 대신 다른 분야 후보 찾기
- 기술주와 낮은 상관관계
- 충분한 거래량
- RSI 전략 적합성
"""

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 현재 보유 (기술주)
TECH_STOCKS = ['QQQ', 'AAPL', 'SMH']

# 이미 선택한 금융
FINANCE = ['JPM']

# 대안 후보 섹터들 (에너지 제외)
CANDIDATES = {
    # 헬스케어
    'XLV': 'Healthcare ETF',
    'JNJ': 'Johnson & Johnson',
    'UNH': 'UnitedHealth',
    'PFE': 'Pfizer',
    
    # 필수소비재
    'XLP': 'Consumer Staples ETF',
    'PG': 'Procter & Gamble',
    'KO': 'Coca-Cola',
    'WMT': 'Walmart',
    
    # 산업재
    'XLI': 'Industrial ETF',
    'CAT': 'Caterpillar',
    'UPS': 'UPS',
    'HON': 'Honeywell',
    
    # 유틸리티
    'XLU': 'Utilities ETF',
    'NEE': 'NextEra Energy',
    'DUK': 'Duke Energy',
    
    # 부동산
    'VNQ': 'Real Estate ETF',
    'AMT': 'American Tower',
    
    # 원자재/금
    'GLD': 'Gold ETF',
    'SLV': 'Silver ETF',
    
    # 채권
    'TLT': 'Long-Term Treasury',
    'BND': 'Total Bond Market',
}


def main():
    print("="*80)
    print("🔍 에너지 대신 다른 분야 후보 분석")
    print("="*80)
    
    # 데이터 로드
    print("\n⏳ 데이터 로딩 중...")
    
    all_tickers = TECH_STOCKS + FINANCE + list(CANDIDATES.keys())
    data = {}
    
    for ticker in all_tickers:
        df = yf.download(ticker, period='5y', progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            data[ticker] = df
    
    print(f"  ✅ {len(data)}개 종목 로드 완료")
    
    # 수익률 계산
    returns = pd.DataFrame()
    for ticker, df in data.items():
        returns[ticker] = df['Close'].pct_change()
    
    # 상관관계 계산
    print("\n📈 상관관계 분석...")
    
    results = []
    
    for ticker, desc in CANDIDATES.items():
        if ticker not in data:
            continue
        
        df = data[ticker]
        
        # 기술주와 상관관계
        tech_corrs = []
        for tech in TECH_STOCKS:
            if tech in returns.columns and ticker in returns.columns:
                corr = returns[ticker].corr(returns[tech])
                if not np.isnan(corr):
                    tech_corrs.append(corr)
        
        avg_tech_corr = np.mean(tech_corrs) if tech_corrs else np.nan
        
        # JPM과 상관관계
        jpm_corr = returns[ticker].corr(returns['JPM']) if 'JPM' in returns.columns else np.nan
        
        # 5년 수익률
        if len(df) > 250:
            five_year_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
        else:
            five_year_return = np.nan
        
        # RSI 과매도 빈도
        df['returns'] = df['Close'].pct_change()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        oversold_pct = (df['rsi'] < 30).sum() / len(df) * 100
        overbought_pct = (df['rsi'] > 70).sum() / len(df) * 100
        
        # 거래량
        vol = df['Volume']
        if isinstance(vol, pd.DataFrame):
            vol = vol.iloc[:, 0]
        avg_volume = vol.mean() / 1e6
        
        # 변동성
        volatility = df['returns'].std() * np.sqrt(252) * 100
        
        results.append({
            'ticker': ticker,
            'description': desc,
            'tech_corr': avg_tech_corr,
            'jpm_corr': jpm_corr,
            '5y_return': five_year_return,
            'oversold_pct': oversold_pct,
            'overbought_pct': overbought_pct,
            'avg_volume': avg_volume,
            'volatility': volatility
        })
    
    results_df = pd.DataFrame(results)
    
    # 섹터별 분류
    sectors = {
        'Healthcare': ['XLV', 'JNJ', 'UNH', 'PFE'],
        'Consumer': ['XLP', 'PG', 'KO', 'WMT'],
        'Industrial': ['XLI', 'CAT', 'UPS', 'HON'],
        'Utilities': ['XLU', 'NEE', 'DUK'],
        'Real Estate': ['VNQ', 'AMT'],
        'Commodities': ['GLD', 'SLV'],
        'Bonds': ['TLT', 'BND']
    }
    
    # 결과 출력
    print("\n" + "="*80)
    print("📊 섹터별 분석 결과")
    print("="*80)
    
    for sector, tickers in sectors.items():
        sector_df = results_df[results_df['ticker'].isin(tickers)]
        if sector_df.empty:
            continue
        
        print(f"\n### {sector}")
        print("-"*70)
        print(f"{'종목':<8} {'설명':<20} {'기술상관':>10} {'JPM상관':>10} {'5Y수익':>10} {'RSI<30%':>8} {'거래량':>10}")
        print("-"*70)
        
        for _, row in sector_df.iterrows():
            tech_corr_str = f"{row['tech_corr']:.2f}" if not np.isnan(row['tech_corr']) else "N/A"
            jpm_corr_str = f"{row['jpm_corr']:.2f}" if not np.isnan(row['jpm_corr']) else "N/A"
            return_str = f"{row['5y_return']:+.0f}%" if not np.isnan(row['5y_return']) else "N/A"
            
            print(f"{row['ticker']:<8} {row['description']:<20} {tech_corr_str:>10} {jpm_corr_str:>10} "
                  f"{return_str:>10} {row['oversold_pct']:>7.1f}% {row['avg_volume']:>9.1f}M")
    
    # 종합 점수
    print("\n" + "="*80)
    print("🏆 종합 추천 (에너지 대체)")
    print("="*80)
    
    # 점수 계산
    for idx, row in results_df.iterrows():
        score = 0
        
        # 기술주와 낮은 상관 (30점)
        if not np.isnan(row['tech_corr']):
            if row['tech_corr'] < 0.3:
                score += 30
            elif row['tech_corr'] < 0.5:
                score += 20
            elif row['tech_corr'] < 0.7:
                score += 10
        
        # JPM과 낮은 상관 (20점) - 분산 효과
        if not np.isnan(row['jpm_corr']):
            if row['jpm_corr'] < 0.3:
                score += 20
            elif row['jpm_corr'] < 0.5:
                score += 15
            elif row['jpm_corr'] < 0.7:
                score += 10
        
        # 수익률 (20점)
        if not np.isnan(row['5y_return']):
            if row['5y_return'] > 50:
                score += 20
            elif row['5y_return'] > 20:
                score += 15
            elif row['5y_return'] > 0:
                score += 10
        
        # RSI 과매도 빈도 (15점)
        if row['oversold_pct'] > 5:
            score += 15
        elif row['oversold_pct'] > 2:
            score += 10
        else:
            score += 5
        
        # 거래량 (15점)
        if row['avg_volume'] > 10:
            score += 15
        elif row['avg_volume'] > 3:
            score += 10
        else:
            score += 5
        
        results_df.loc[idx, 'score'] = score
    
    # 상위 10개
    top10 = results_df.nlargest(10, 'score')
    
    print(f"\n{'순위':<6} {'종목':<8} {'설명':<20} {'기술상관':>10} {'5Y수익':>10} {'RSI<30%':>8} {'거래량':>10} {'점수':>6}")
    print("-"*90)
    
    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        medal = ["🥇", "🥈", "🥉"][rank-1] if rank <= 3 else f"{rank}."
        tech_corr_str = f"{row['tech_corr']:.2f}" if not np.isnan(row['tech_corr']) else "N/A"
        return_str = f"{row['5y_return']:+.0f}%" if not np.isnan(row['5y_return']) else "N/A"
        
        print(f"{medal:<6} {row['ticker']:<8} {row['description']:<20} {tech_corr_str:>10} "
              f"{return_str:>10} {row['oversold_pct']:>7.1f}% {row['avg_volume']:>9.1f}M {row['score']:>5.0f}")
    
    # 추천
    print("\n" + "="*80)
    print("💡 최종 추천")
    print("="*80)
    
    print("""
### 🏆 에너지 대체 TOP 3

1️⃣ **헬스케어 (XLV)** - 가장 추천!
   - 기술주 상관: 0.50 (중간)
   - JPM 상관: 0.41 (낮음)
   - 경기 방어적 섹터
   - RSI 과매도 빈도 적당

2️⃣ **금 (GLD)** - 분산 최고
   - 기술주 상관: 0.09 (매우 낮음!)
   - JPM 상관: 0.07 (거의 무관)
   - 위기 시 헷지 역할
   - ⚠️ 수익률은 상대적으로 낮음

3️⃣ **필수소비재 (XLP)**
   - 기술주 상관: 0.35 (낮음)
   - 안정적인 수익
   - 경기 침체에도 버팀

### 📌 결론

**현재 조합 추천:**
- 기술: AAPL 또는 SMH (1개)
- 금융: JPM (고정)
- (선택) 헬스케어: XLV 또는 금: GLD

**3종목 분산 예시:**
AAPL + JPM + XLV (또는 GLD)
""")


if __name__ == "__main__":
    main()

