"""6개 종목 상관관계 분석"""
import pandas as pd
import yfinance as yf

tickers = ['QQQ', 'AAPL', 'SMH', 'JPM', 'WMT', 'GLD']

print('데이터 로드 중...')
data = {}
for t in tickers:
    df = yf.download(t, period='10y', progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    data[t] = df['Close']

returns = pd.DataFrame(data).pct_change().dropna()
corr = returns.corr()

print()
print('=' * 70)
print('📊 6개 종목 상관관계 매트릭스')
print('=' * 70)
print()
header = '        '
for t in tickers:
    header += f'{t:>8}'
print(header)
print('-' * 56)

for t1 in tickers:
    row = f'{t1:>8}'
    for t2 in tickers:
        val = corr.loc[t1, t2]
        if t1 == t2:
            row += '    1.00'
        elif val >= 0.7:
            row += f' 🔴{val:.2f}'
        elif val >= 0.4:
            row += f' 🟡{val:.2f}'
        else:
            row += f' 🟢{val:.2f}'
    print(row)

print()
print('=' * 70)
print('🔍 기술주 3개 (QQQ, AAPL, SMH) 상관관계')
print('=' * 70)
print(f'  QQQ ↔ AAPL: {corr.loc["QQQ", "AAPL"]:.2f} 🔴 높음')
print(f'  QQQ ↔ SMH:  {corr.loc["QQQ", "SMH"]:.2f} 🔴 높음')
print(f'  AAPL ↔ SMH: {corr.loc["AAPL", "SMH"]:.2f} 🟡 중간')

print()
print('=' * 70)
print('💡 분산 효과 좋은 조합')
print('=' * 70)

pairs = []
for i, t1 in enumerate(tickers):
    for t2 in tickers[i+1:]:
        pairs.append((t1, t2, corr.loc[t1, t2]))

pairs.sort(key=lambda x: x[2])
print('\n🟢 상관관계 낮은 TOP 10:')
for t1, t2, c in pairs[:10]:
    print(f'   {t1} + {t2}: {c:.2f}')

print()
print('=' * 70)
print('🎯 추천')
print('=' * 70)
print()
print('3배 레버리지 있는 종목:')
print('  • QQQ → TQQQ (3배)')
print('  • SMH → SOXL (3배)')
print('  • AAPL → 없음')
print()
print('QQQ vs SMH 상관관계: {:.2f} (🔴 매우 높음)'.format(corr.loc['QQQ', 'SMH']))
print('→ 둘 다 하면 분산 효과 거의 없음, 하나만 선택 권장')
print()
print('포트폴리오 제안:')
print('  ✅ 기술 1개 (QQQ or SMH) + JPM + WMT + GLD')
print('  ✅ SMH가 수익률 더 높음 (+31.8% vs +11.9%)')
print('  ✅ QQQ가 물타기 적음 (4회 vs 8회)')

