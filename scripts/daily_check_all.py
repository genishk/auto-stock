"""
통합 일일 시그널 체크 스크립트
6개 종목 (QQQ, AAPL, SMH, JPM, WMT, GLD)을 한 번에 체크하고
하나의 종합 리포트로 출력

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
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ===== 6개 종목 전략 파라미터 =====
STRATEGIES = {
    'QQQ': {
        'icon': '📊',
        'name': 'QQQ (나스닥100)',
        'RSI_OVERSOLD': 35,
        'RSI_BUY_EXIT': 55,
        'RSI_OVERBOUGHT': 60,
        'RSI_SELL_EXIT': 45,
    },
    'AAPL': {
        'icon': '🍎',
        'name': 'AAPL (애플)',
        'RSI_OVERSOLD': 35,
        'RSI_BUY_EXIT': 40,
        'RSI_OVERBOUGHT': 60,
        'RSI_SELL_EXIT': 45,
    },
    'SMH': {
        'icon': '💎',
        'name': 'SMH (반도체)',
        'RSI_OVERSOLD': 35,
        'RSI_BUY_EXIT': 40,
        'RSI_OVERBOUGHT': 75,
        'RSI_SELL_EXIT': 45,
    },
    'JPM': {
        'icon': '🏦',
        'name': 'JPM (JP모건)',
        'RSI_OVERSOLD': 40,
        'RSI_BUY_EXIT': 55,
        'RSI_OVERBOUGHT': 60,
        'RSI_SELL_EXIT': 45,
    },
    'WMT': {
        'icon': '🏪',
        'name': 'WMT (월마트)',
        'RSI_OVERSOLD': 45,
        'RSI_BUY_EXIT': 55,
        'RSI_OVERBOUGHT': 60,
        'RSI_SELL_EXIT': 55,
    },
    'GLD': {
        'icon': '🥇',
        'name': 'GLD (금)',
        'RSI_OVERSOLD': 40,
        'RSI_BUY_EXIT': 50,
        'RSI_OVERBOUGHT': 65,
        'RSI_SELL_EXIT': 60,
    },
}

CAPITAL_PER_ENTRY = 1000


def find_buy_signals(df, params):
    """매수 시그널 찾기"""
    buy_signals = []
    in_oversold = False
    last_signal_date = None
    last_signal_price = None
    
    RSI_OVERSOLD = params['RSI_OVERSOLD']
    RSI_BUY_EXIT = params['RSI_BUY_EXIT']
    
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


def find_sell_signals(df, params):
    """매도 시그널 찾기"""
    sell_signals = []
    in_overbought = False
    last_signal_date = None
    last_signal_price = None
    
    RSI_OVERBOUGHT = params['RSI_OVERBOUGHT']
    RSI_SELL_EXIT = params['RSI_SELL_EXIT']
    
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
    """거래 시뮬레이션 - 동일 금액, profit_only"""
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


def analyze_ticker(ticker, params, config, cache):
    """단일 종목 분석"""
    # 데이터 로드
    df = cache.get(ticker)
    if df is None:
        fetcher = DataFetcher([ticker])
        data = fetcher.fetch('10y')
        if ticker not in data:
            return None
        df = data[ticker]
        df, _ = DataValidator.validate(df, ticker)
        cache.set(ticker, df)
    
    # 기술 지표 계산
    ti = TechnicalIndicators(config.get('indicators', {}))
    df = ti.calculate_all(df)
    
    # 이동평균선
    df['MA40'] = df['Close'].rolling(window=40).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['golden_cross'] = df['MA40'] > df['MA200']
    
    # 최신 데이터
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    current_date = df.index[-1]
    current_rsi = latest.get('rsi', 0)
    current_price = latest['Close']
    price_change = (current_price / prev['Close'] - 1) * 100
    
    # 시그널 및 거래 시뮬레이션
    buy_signals = find_buy_signals(df, params)
    sell_signals = find_sell_signals(df, params)
    trades, positions = simulate_trades(df, buy_signals, sell_signals)
    
    # 오늘 시그널 확인
    today = df.index[-1]
    buy_signal = any(bs['confirm_date'] == today for bs in buy_signals)
    sell_signal = any(ss['confirm_date'] == today for ss in sell_signals)
    
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
    
    # 액션 판단
    action = 'none'
    action_detail = ''
    action_emoji = ''
    
    if buy_signal:
        if has_position:
            action = 'add'
            action_detail = f'물타기 ({position_count}→{position_count+1}회)'
            action_emoji = '🔵'
        else:
            action = 'buy'
            action_detail = '신규 매수'
            action_emoji = '🟢'
    elif sell_signal:
        if has_position:
            if unrealized_pct > 0:
                action = 'sell'
                action_detail = f'익절 ({unrealized_pct:+.1f}%)'
                action_emoji = '💰'
            else:
                action = 'hold'
                action_detail = f'손실 중 ({unrealized_pct:+.1f}%) → 홀드'
                action_emoji = '⏸️'
        else:
            action = 'skip'
            action_detail = '포지션 없음 → 무시'
            action_emoji = '⏭️'
    else:
        if has_position:
            action_detail = '홀드'
            action_emoji = '📦'
        else:
            action_detail = '대기'
            action_emoji = '⏳'
    
    # 시그널 타입
    if buy_signal:
        signal_type = 'buy'
        signal_emoji = '🟢'
    elif sell_signal:
        signal_type = 'sell'
        signal_emoji = '🔴'
    else:
        signal_type = 'none'
        signal_emoji = '⚪'
    
    return {
        'ticker': ticker,
        'icon': params['icon'],
        'name': params['name'],
        'price': current_price,
        'price_change': price_change,
        'rsi': current_rsi,
        'signal_type': signal_type,
        'signal_emoji': signal_emoji,
        'action': action,
        'action_detail': action_detail,
        'action_emoji': action_emoji,
        'has_position': has_position,
        'position_count': position_count,
        'avg_price': avg_price,
        'unrealized_pct': unrealized_pct,
        'total_invested': total_invested,
        'params': params,
    }


def main():
    config = load_config()
    cache = DataCache(cache_dir='data/cache', max_age_hours=24)
    
    # 오늘 날짜
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    # 모든 종목 분석
    results = []
    for ticker, params in STRATEGIES.items():
        print(f'Analyzing {ticker}...')
        result = analyze_ticker(ticker, params, config, cache)
        if result:
            results.append(result)
    
    # 결과 분류
    action_required = [r for r in results if r['action'] in ['buy', 'add', 'sell']]
    signals_only = [r for r in results if r['signal_type'] != 'none' and r['action'] not in ['buy', 'add', 'sell']]
    no_signal = [r for r in results if r['signal_type'] == 'none']
    
    # 콘솔 출력
    print()
    print('=' * 60)
    print('📊 Auto-Stock 통합 일일 리포트')
    print('=' * 60)
    print(f'📅 날짜: {current_date}')
    print()
    
    # 액션 필요한 종목
    if action_required:
        print('🚨 액션 필요!')
        print('-' * 40)
        for r in action_required:
            print(f"{r['icon']} {r['ticker']}: {r['action_emoji']} {r['action_detail']} @ ${r['price']:.2f}")
        print()
    
    # 시그널만 있는 종목 (액션 없음)
    if signals_only:
        print('📡 시그널 발생 (액션 없음)')
        print('-' * 40)
        for r in signals_only:
            print(f"{r['icon']} {r['ticker']}: {r['signal_emoji']} {'매수' if r['signal_type'] == 'buy' else '매도'} 시그널 → {r['action_detail']}")
        print()
    
    # 모든 종목 현황
    print('📋 전체 현황')
    print('-' * 40)
    for r in results:
        pos_str = f"보유 {r['position_count']}회 ({r['unrealized_pct']:+.1f}%)" if r['has_position'] else "대기"
        print(f"{r['icon']} {r['ticker']}: ${r['price']:.2f} ({r['price_change']:+.1f}%) | RSI {r['rsi']:.1f} | {pos_str}")
    
    print()
    print('=' * 60)
    
    # GitHub Actions 환경 변수
    github_output = os.environ.get('GITHUB_OUTPUT', '')
    if github_output:
        with open(github_output, 'a') as f:
            f.write(f'current_date={current_date}\n')
            f.write(f'total_tickers={len(results)}\n')
            f.write(f'action_count={len(action_required)}\n')
            f.write(f'signal_count={len(signals_only)}\n')
            
            # 액션 필요 여부
            has_action = len(action_required) > 0
            f.write(f'has_action={"yes" if has_action else "no"}\n')
            
            # 제목용 요약
            if action_required:
                actions = [f"{r['ticker']} {r['action_emoji']}" for r in action_required]
                f.write(f'subject_summary=🚨 {", ".join(actions)}\n')
            elif signals_only:
                f.write(f'subject_summary=📡 시그널 {len(signals_only)}개 (액션 없음)\n')
            else:
                f.write(f'subject_summary=✅ 시그널 없음\n')
    
    # 이메일 본문 파일로 저장
    email_body = []
    email_body.append("═══════════════════════════════════════")
    email_body.append("📊 Auto-Stock 통합 일일 리포트")
    email_body.append("═══════════════════════════════════════")
    email_body.append("")
    email_body.append(f"📅 날짜: {current_date}")
    email_body.append("")
    
    # 액션 필요 섹션
    if action_required:
        email_body.append("🚨 액션 필요!")
        email_body.append("─────────────────────────────────────")
        for r in action_required:
            email_body.append(f"{r['icon']} {r['ticker']}: {r['action_emoji']} {r['action_detail']}")
            email_body.append(f"   가격: ${r['price']:.2f} ({r['price_change']:+.1f}%) | RSI: {r['rsi']:.1f}")
            if r['has_position']:
                email_body.append(f"   포지션: {r['position_count']}회 물타기 | 평단가: ${r['avg_price']:.2f} | 손익: {r['unrealized_pct']:+.1f}%")
            email_body.append("")
    else:
        email_body.append("✅ 오늘 액션 필요 없음")
        email_body.append("")
    
    # 시그널만 섹션
    if signals_only:
        email_body.append("📡 시그널 발생 (액션 없음)")
        email_body.append("─────────────────────────────────────")
        for r in signals_only:
            sig = '매수' if r['signal_type'] == 'buy' else '매도'
            email_body.append(f"{r['icon']} {r['ticker']}: {r['signal_emoji']} {sig} 시그널 → {r['action_detail']}")
        email_body.append("")
    
    # 전체 현황
    email_body.append("📋 전체 현황")
    email_body.append("─────────────────────────────────────")
    for r in results:
        pos_str = f"보유 {r['position_count']}회 ({r['unrealized_pct']:+.1f}%)" if r['has_position'] else "대기"
        email_body.append(f"{r['icon']} {r['ticker']}: ${r['price']:.2f} ({r['price_change']:+.1f}%) | RSI {r['rsi']:.1f} | {pos_str}")
    email_body.append("")
    
    # 전략 기준
    email_body.append("📊 전략 기준")
    email_body.append("─────────────────────────────────────")
    for ticker, params in STRATEGIES.items():
        email_body.append(f"{params['icon']} {ticker}: RSI {params['RSI_OVERSOLD']}/{params['RSI_BUY_EXIT']} → {params['RSI_OVERBOUGHT']}/{params['RSI_SELL_EXIT']}")
    email_body.append("")
    email_body.append("═══════════════════════════════════════")
    
    # 이메일 본문 문자열
    email_body_str = '\n'.join(email_body)
    
    # 파일로도 저장 (디버깅용)
    with open('email_body.txt', 'w', encoding='utf-8') as f:
        f.write(email_body_str)
    
    # 이메일 제목 생성
    if action_required:
        actions = [f"{r['ticker']} {r['action_emoji']}" for r in action_required]
        subject_summary = f"🚨 {', '.join(actions)}"
    elif signals_only:
        subject_summary = f"📡 시그널 {len(signals_only)}개 (액션 없음)"
    else:
        subject_summary = "✅ 시그널 없음"
    
    email_subject = f"Auto-Stock {subject_summary} ({current_date})"
    
    # 환경변수에서 이메일 정보 가져오기
    email_username = os.environ.get('EMAIL_USERNAME', '')
    email_password = os.environ.get('EMAIL_PASSWORD', '')
    email_to = os.environ.get('EMAIL_TO', '')
    
    if email_username and email_password and email_to:
        try:
            # 이메일 전송
            msg = MIMEMultipart()
            msg['From'] = f'Auto-Stock 통합 <{email_username}>'
            msg['To'] = email_to
            msg['Subject'] = email_subject
            msg.attach(MIMEText(email_body_str, 'plain', 'utf-8'))
            
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(email_username, email_password)
            server.sendmail(email_username, email_to, msg.as_string())
            server.quit()
            
            print(f'✅ 이메일 전송 완료: {email_to}')
        except Exception as e:
            print(f'❌ 이메일 전송 실패: {e}')
    else:
        print('⚠️ 이메일 환경변수 미설정 (EMAIL_USERNAME, EMAIL_PASSWORD, EMAIL_TO)')


if __name__ == '__main__':
    main()

