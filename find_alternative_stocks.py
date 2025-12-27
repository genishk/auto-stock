"""
현재 종목(QQQ, AAPL, SMH)과 상관관계 낮은 대안 종목 탐색
- 섹터 분산 (기술주 외)
- RSI 전략 적합성 (변동성 적절)
- 거래량 활발
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 현재 보유 종목
CURRENT_TICKERS = ['QQQ', 'AAPL', 'SMH']

# 검토할 대안 종목들 (섹터별)
CANDIDATE_TICKERS = {
    # 금융 섹터
    'XLF': '금융 섹터 ETF',
    'JPM': 'JP모건 (은행)',
    'GS': '골드만삭스',
    'BRK-B': '버크셔해서웨이',
    
    # 에너지/원자재
    'XLE': '에너지 섹터 ETF',
    'XOM': '엑손모빌 (석유)',
    'CVX': '셰브론 (석유)',
    'GLD': '금 ETF',
    'USO': '원유 ETF',
    
    # 헬스케어
    'XLV': '헬스케어 섹터 ETF',
    'JNJ': '존슨앤존슨',
    'UNH': '유나이티드헬스',
    'PFE': '화이자',
    
    # 소비재/필수소비재
    'XLP': '필수소비재 ETF',
    'XLY': '경기소비재 ETF',
    'COST': '코스트코',
    'WMT': '월마트',
    'MCD': '맥도날드',
    
    # 산업재/유틸리티
    'XLI': '산업재 ETF',
    'XLU': '유틸리티 ETF',
    'CAT': '캐터필러',
    'UPS': 'UPS',
    
    # 리츠
    'VNQ': '리츠 ETF',
    'O': '리얼티인컴',
    
    # 배당주
    'VYM': '고배당 ETF',
    'SCHD': '배당성장 ETF',
    
    # 가치주
    'VTV': '가치주 ETF',
    'IWD': '러셀 가치 ETF',
    
    # 소형주
    'IWM': '러셀2000 (소형주)',
    
    # 신흥국
    'EEM': '신흥국 ETF',
    'VWO': '신흥국 ETF (Vanguard)',
    
    # 채권
    'TLT': '장기국채 ETF',
    'BND': '총채권 ETF',
}


def get_data(tickers: list, period: str = '5y'):
    """여러 종목 데이터 가져오기"""
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, progress=False)
            if len(df) > 100:
                data[ticker] = df
        except Exception as e:
            print(f"  ⚠️ {ticker} 로드 실패: {e}")
    return data


def calculate_rsi(prices: pd.Series, period: int = 14):
    """RSI 계산"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def analyze_candidate(ticker: str, df: pd.DataFrame, current_returns: pd.DataFrame):
    """후보 종목 분석"""
    # 일간 수익률 - Series로 변환
    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    returns = close.pct_change().dropna()
    
    # 현재 종목들과 상관관계
    correlations = {}
    for curr_ticker in CURRENT_TICKERS:
        if curr_ticker in current_returns.columns:
            curr_ret = current_returns[curr_ticker].dropna()
            # 공통 날짜만
            common_idx = returns.index.intersection(curr_ret.index)
            if len(common_idx) > 100:
                corr = returns.loc[common_idx].corr(curr_ret.loc[common_idx])
                correlations[curr_ticker] = float(corr)
    
    avg_corr = np.mean(list(correlations.values())) if correlations else 0
    
    # RSI 계산
    rsi = calculate_rsi(close)
    
    # RSI 과매도 발생 빈도 (RSI < 35)
    oversold_count = (rsi < 35).sum()
    oversold_pct = oversold_count / len(rsi) * 100
    
    # RSI 과매수 발생 빈도 (RSI > 70)
    overbought_count = (rsi > 70).sum()
    overbought_pct = overbought_count / len(rsi) * 100
    
    # 변동성 (연간)
    volatility = returns.std() * np.sqrt(252) * 100
    
    # 평균 거래량 (백만)
    volume = df['Volume']
    if isinstance(volume, pd.DataFrame):
        volume = volume.iloc[:, 0]
    avg_volume = volume.mean() / 1e6
    
    # 5년 수익률
    total_return = (close.iloc[-1] / close.iloc[0] - 1) * 100
    
    # RSI 전략 적합성 점수
    # - 과매도 5-15% 정도가 이상적 (너무 많으면 하락 추세, 너무 적으면 기회 없음)
    if 5 <= oversold_pct <= 15:
        rsi_score = 10
    elif 3 <= oversold_pct <= 20:
        rsi_score = 7
    else:
        rsi_score = 3
    
    return {
        'ticker': ticker,
        'correlations': correlations,
        'avg_correlation': avg_corr,
        'volatility': volatility,
        'avg_volume_M': avg_volume,
        'total_return_5y': total_return,
        'oversold_pct': oversold_pct,
        'overbought_pct': overbought_pct,
        'rsi_score': rsi_score
    }


def main():
    print("="*70)
    print("🔍 대안 종목 탐색")
    print("="*70)
    print(f"현재 보유: {', '.join(CURRENT_TICKERS)}")
    print(f"검토 종목: {len(CANDIDATE_TICKERS)}개")
    
    # 현재 종목 데이터 로드
    print("\n⏳ 현재 종목 데이터 로딩...")
    current_data = get_data(CURRENT_TICKERS, period='5y')
    
    # 현재 종목 수익률
    current_returns = pd.DataFrame()
    for ticker, df in current_data.items():
        close = df['Close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        current_returns[ticker] = close.pct_change()
    
    print(f"✅ {len(current_data)}개 현재 종목 로드 완료")
    
    # 후보 종목 분석
    print("\n⏳ 후보 종목 분석 중...")
    candidates = []
    
    all_tickers = list(CANDIDATE_TICKERS.keys())
    candidate_data = get_data(all_tickers, period='5y')
    
    for ticker, desc in CANDIDATE_TICKERS.items():
        if ticker in candidate_data:
            result = analyze_candidate(ticker, candidate_data[ticker], current_returns)
            result['description'] = desc
            candidates.append(result)
    
    print(f"✅ {len(candidates)}개 후보 종목 분석 완료")
    
    # ===== 결과 출력 =====
    print("\n" + "="*70)
    print("📊 상관관계 분석 결과")
    print("="*70)
    
    # 상관관계 낮은 순 정렬
    candidates.sort(key=lambda x: x['avg_correlation'])
    
    print(f"\n{'종목':<8} {'설명':<20} {'QQQ':>6} {'AAPL':>6} {'SMH':>6} {'평균':>6} {'변동성':>8}")
    print("-" * 70)
    
    for c in candidates[:20]:  # 상위 20개
        corrs = c['correlations']
        print(f"{c['ticker']:<8} {c['description']:<20} "
              f"{corrs.get('QQQ', 0):>5.2f} {corrs.get('AAPL', 0):>5.2f} {corrs.get('SMH', 0):>5.2f} "
              f"{c['avg_correlation']:>5.2f} {c['volatility']:>7.1f}%")
    
    # ===== RSI 전략 적합성 =====
    print("\n" + "="*70)
    print("📈 RSI 전략 적합성 (상관관계 0.6 이하)")
    print("="*70)
    
    # 상관관계 낮고 RSI 전략 적합한 종목
    good_candidates = [c for c in candidates if c['avg_correlation'] < 0.6]
    good_candidates.sort(key=lambda x: (x['rsi_score'], -x['avg_correlation']), reverse=True)
    
    print(f"\n{'종목':<8} {'설명':<18} {'과매도%':>8} {'과매수%':>8} {'5년수익':>10} {'거래량M':>8} {'상관':>6} {'점수':>6}")
    print("-" * 80)
    
    for c in good_candidates[:15]:
        print(f"{c['ticker']:<8} {c['description']:<18} "
              f"{c['oversold_pct']:>7.1f}% {c['overbought_pct']:>7.1f}% "
              f"{c['total_return_5y']:>+9.1f}% {c['avg_volume_M']:>7.1f} "
              f"{c['avg_correlation']:>5.2f} {c['rsi_score']:>5}")
    
    # ===== 최종 추천 =====
    print("\n" + "="*70)
    print("🏆 최종 추천")
    print("="*70)
    
    # 종합 점수 계산
    for c in candidates:
        # 상관관계 점수 (낮을수록 좋음) - 30점
        corr_score = max(0, 30 - c['avg_correlation'] * 40)
        
        # RSI 적합성 - 20점
        rsi_score = c['rsi_score'] * 2
        
        # 수익률 점수 - 20점
        return_score = min(20, max(0, c['total_return_5y'] / 5))
        
        # 거래량 점수 - 15점
        vol_score = min(15, c['avg_volume_M'] / 10)
        
        # 변동성 점수 (적당한 변동성이 좋음, 20-40% 이상적) - 15점
        if 20 <= c['volatility'] <= 40:
            volatility_score = 15
        elif 15 <= c['volatility'] <= 50:
            volatility_score = 10
        else:
            volatility_score = 5
        
        c['total_score'] = corr_score + rsi_score + return_score + vol_score + volatility_score
    
    # 점수순 정렬
    candidates.sort(key=lambda x: x['total_score'], reverse=True)
    
    print("\n### 🎯 TOP 10 추천 종목")
    print("-" * 70)
    print(f"{'순위':<4} {'종목':<8} {'설명':<20} {'상관':>6} {'수익률':>10} {'점수':>8}")
    print("-" * 70)
    
    for i, c in enumerate(candidates[:10]):
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."
        print(f"{medal:<4} {c['ticker']:<8} {c['description']:<20} "
              f"{c['avg_correlation']:>5.2f} {c['total_return_5y']:>+9.1f}% {c['total_score']:>7.1f}")
    
    # 추천 이유
    print("\n### 💡 TOP 3 상세 분석")
    print("-" * 70)
    
    for i, c in enumerate(candidates[:3]):
        print(f"\n{['🥇', '🥈', '🥉'][i]} **{c['ticker']}** ({c['description']})")
        print(f"  - 상관관계: QQQ {c['correlations'].get('QQQ', 0):.2f}, "
              f"AAPL {c['correlations'].get('AAPL', 0):.2f}, SMH {c['correlations'].get('SMH', 0):.2f}")
        print(f"  - 5년 수익률: {c['total_return_5y']:+.1f}%")
        print(f"  - 변동성: {c['volatility']:.1f}% (연간)")
        print(f"  - RSI 과매도 빈도: {c['oversold_pct']:.1f}%")
        print(f"  - 평균 거래량: {c['avg_volume_M']:.1f}M/일")
        
        # 장점/단점
        pros = []
        cons = []
        
        if c['avg_correlation'] < 0.3:
            pros.append("매우 낮은 상관관계")
        elif c['avg_correlation'] < 0.5:
            pros.append("낮은 상관관계")
        
        if c['total_return_5y'] > 50:
            pros.append("높은 장기 수익률")
        elif c['total_return_5y'] < 0:
            cons.append("마이너스 장기 수익률")
        
        if 5 <= c['oversold_pct'] <= 15:
            pros.append("RSI 전략 적합")
        elif c['oversold_pct'] < 3:
            cons.append("과매도 기회 적음")
        
        if c['avg_volume_M'] > 50:
            pros.append("높은 유동성")
        elif c['avg_volume_M'] < 5:
            cons.append("낮은 유동성")
        
        print(f"  ✅ 장점: {', '.join(pros) if pros else '없음'}")
        print(f"  ⚠️ 단점: {', '.join(cons) if cons else '없음'}")


if __name__ == "__main__":
    main()

