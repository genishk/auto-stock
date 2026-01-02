"""
MA (Mastercard) 물타기 전략 대시보드
streamlit run dashboard_ma.py --server.port 8508

고빈도 전략: RSI 45/50 → 60/55, GC OFF
- 거래 28회, 물타기 최대 7회, 승률 100%, 수익률 +6.1%
- 금융 섹터 (결제 네트워크)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.cache import DataCache
from src.data.fetcher import DataFetcher
from src.data.validator import DataValidator
from src.features.technical import TechnicalIndicators
from src.utils.helpers import load_config

# ===== MA 전략 파라미터 =====
TICKER = "MA"
RSI_OVERSOLD = 45
RSI_BUY_EXIT = 50
RSI_OVERBOUGHT = 60
RSI_SELL_EXIT = 55
USE_GOLDEN_CROSS = False
CAPITAL_PER_ENTRY = 1000

# 페이지 설정
st.set_page_config(
    page_title="MA 물타기 전략",
    page_icon="💳",
    layout="wide"
)


@st.cache_data(ttl=3600)
def load_data():
    """데이터 로드 및 지표 계산"""
    config = load_config()
    cache = DataCache(
        cache_dir=str(project_root / config['data']['cache']['directory']),
        max_age_hours=24
    )
    
    df = cache.get(TICKER)
    if df is None:
        fetcher = DataFetcher([TICKER])
        data = fetcher.fetch('10y')
        if TICKER in data:
            df = data[TICKER]
            df, _ = DataValidator.validate(df, TICKER)
            cache.set(TICKER, df)
    
    if df is not None:
        indicators = TechnicalIndicators(config.get('indicators', {}))
        df = indicators.calculate_all(df)
        
        # 골든크로스용 이동평균선
        df['MA40'] = df['Close'].rolling(window=40).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        df['golden_cross'] = df['MA40'] > df['MA200']
    
    return df


def find_buy_signals(df):
    """매수 시그널 찾기"""
    buy_signals = []
    in_oversold = False
    last_signal_date = None
    last_signal_price = None
    
    for idx in range(len(df)):
        rsi = df['rsi'].iloc[idx]
        if pd.isna(rsi):
            continue
        
        golden_cross_ok = True
        if USE_GOLDEN_CROSS and 'golden_cross' in df.columns:
            gc = df['golden_cross'].iloc[idx]
            golden_cross_ok = gc if not pd.isna(gc) else False
        
        if rsi < RSI_OVERSOLD:
            in_oversold = True
            last_signal_date = df.index[idx]
            last_signal_price = df['Close'].iloc[idx]
        else:
            if in_oversold and rsi >= RSI_BUY_EXIT and last_signal_date is not None:
                if golden_cross_ok:
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
    """매도 시그널 찾기"""
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
    """거래 시뮬레이션 (동일 금액, 최소 수익률 2% 이상)"""
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
                if sell_return >= 2:  # 최소 수익률 2% 이상
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
    st.title(f"💳 {TICKER} 물타기 전략")
    st.caption(f"금융 섹터 (결제 네트워크) | 마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # 사이드바
    st.sidebar.header("⚙️ 전략 설정")
    st.sidebar.info(f"""
    **{TICKER} 고빈도 전략**
    - 매수: RSI < {RSI_OVERSOLD} → ≥ {RSI_BUY_EXIT}
    - 매도: RSI > {RSI_OVERBOUGHT} → ≤ {RSI_SELL_EXIT}
    - 골든크로스: {'ON' if USE_GOLDEN_CROSS else 'OFF'}
    - **최소 수익률 2% 이상일 때 익절**
    
    *10년 백테스트: 26회 거래, 승률 100%, +7.4%*
    """)
    
    lookback_days = st.sidebar.slider("표시 기간 (일)", 30, 3650, 365)
    
    # 데이터 로드
    df = load_data()
    
    if df is None:
        st.error(f"❌ {TICKER} 데이터를 불러올 수 없습니다.")
        return
    
    st.sidebar.success(f"✅ {len(df)}일 데이터 로드")
    st.sidebar.info(f"📅 {df.index[0].date()} ~ {df.index[-1].date()}")
    
    # 시그널 및 거래 계산
    buy_signals = find_buy_signals(df)
    sell_signals = find_sell_signals(df)
    trades, positions = simulate_trades(df, buy_signals, sell_signals)
    
    # 탭 구성
    tab1, tab2, tab3 = st.tabs(["📊 현재 상태", "📈 통합 뷰", "📋 전체 성과"])
    
    # ===== 탭 1: 현재 상태 =====
    with tab1:
        st.header(f"📊 {TICKER} 현재 상태")
        
        current = df['Close'].iloc[-1]
        prev = df['Close'].iloc[-2]
        change = (current / prev - 1) * 100
        rsi_now = df['rsi'].iloc[-1]
        current_gc = df['golden_cross'].iloc[-1] if 'golden_cross' in df.columns else False
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("현재가", f"${current:.2f}", f"{change:+.2f}%")
        with col2:
            rsi_status = "🔴 과매도" if rsi_now < RSI_OVERSOLD else ("🟢 과매수" if rsi_now > RSI_OVERBOUGHT else "⚪ 중립")
            st.metric("RSI", f"{rsi_now:.1f}", delta=rsi_status)
        with col3:
            gc_status = "🟢 상승장" if current_gc else "🔴 하락장"
            st.metric("추세 (MA40/200)", gc_status)
        with col4:
            if positions:
                n = len(positions)
                total_inv = n * CAPITAL_PER_ENTRY
                total_qty = sum(CAPITAL_PER_ENTRY / p['price'] for p in positions)
                avg_p = total_inv / total_qty
                unrealized = (current / avg_p - 1) * 100
                unrealized_amt = total_inv * unrealized / 100
                st.metric("보유 상태", f"{n}회 물타기 (${total_inv:,})", delta=f"${unrealized_amt:+,.0f} ({unrealized:+.1f}%)")
            else:
                st.metric("보유 상태", "대기 중")
        with col5:
            if trades:
                win_rate = len([t for t in trades if t['return'] > 0]) / len(trades) * 100
                st.metric("전체 승률", f"{win_rate:.0f}%")
        
        st.divider()
        
        # 현재 포지션 상세
        if positions:
            st.subheader("💰 현재 보유 포지션")
            
            n = len(positions)
            total_inv = n * CAPITAL_PER_ENTRY
            total_qty = sum(CAPITAL_PER_ENTRY / p['price'] for p in positions)
            avg_price = total_inv / total_qty
            unrealized_pct = (current / avg_price - 1) * 100
            unrealized_amt = total_inv * unrealized_pct / 100
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("평균 매수가", f"${avg_price:.2f}")
            with col2:
                st.metric("물타기 횟수", f"{n}회")
            with col3:
                st.metric("총 투자금", f"${total_inv:,}")
            with col4:
                color = "🟢" if unrealized_pct >= 0 else "🔴"
                st.metric("미실현 손익", f"{color} ${unrealized_amt:+,.0f} ({unrealized_pct:+.1f}%)")
            
            pos_df = pd.DataFrame([{
                '매수일': p['date'].strftime('%Y-%m-%d'),
                '매수가': f"${p['price']:.2f}",
                '투자금': f"${CAPITAL_PER_ENTRY:,}",
                '현재 손익': f"${CAPITAL_PER_ENTRY * (current/p['price']-1):+,.0f} ({(current/p['price']-1)*100:+.1f}%)"
            } for p in positions])
            st.dataframe(pos_df, use_container_width=True, hide_index=True)
        else:
            st.subheader("⏳ 대기 중")
            st.info("현재 보유 포지션이 없습니다. 매수 시그널 대기 중...")
        
        st.divider()
        
        # 전략 기준 안내
        st.info(f"""
        **💳 {TICKER} 고빈도 전략 (10년 백테스트)**
        
        **📥 매수 조건:** RSI < {RSI_OVERSOLD} 진입 → RSI ≥ {RSI_BUY_EXIT} 탈출 시 매수
        **📤 매도 조건:** RSI > {RSI_OVERBOUGHT} 진입 → RSI ≤ {RSI_SELL_EXIT} 탈출 + **수익률 ≥ 2%** 일 때 매도
        **🛡️ 손절:** 없음 (10년간 승률 100%)
        **📈 골든크로스:** {'✅ 적용중' if USE_GOLDEN_CROSS else '❌ 미적용'}
        
        *성과: 거래 26회, 물타기 최대 7회, 수익률 +7.4%*
        """)
        
        # 가격 차트 + 거래 액션
        st.subheader("📊 가격 차트 + 거래 액션")
        signal_cutoff = df.index[-1] - pd.Timedelta(days=lookback_days)
        chart_df = df[df.index >= signal_cutoff]
        
        # 액션 리스트 생성 (차트용)
        chart_actions = []
        for trade in trades:
            chart_actions.append({
                'date': trade['entry_dates'][0],
                'action': '🟢 매수',
                'price': trade['entry_prices'][0]
            })
            for i in range(1, trade['num_buys']):
                chart_actions.append({
                    'date': trade['entry_dates'][i],
                    'action': f'🔵 물타기 ({i+1}회)',
                    'price': trade['entry_prices'][i]
                })
            chart_actions.append({
                'date': trade['exit_date'],
                'action': '💰 익절',
                'price': trade['exit_price'],
                'return': trade['return']
            })
        
        for i, p in enumerate(positions):
            if i == 0:
                chart_actions.append({
                    'date': p['date'],
                    'action': '🟢 매수 (보유중)',
                    'price': p['price']
                })
            else:
                chart_actions.append({
                    'date': p['date'],
                    'action': f'🔵 물타기 ({i+1}회, 보유중)',
                    'price': p['price']
                })
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=chart_df.index,
            open=chart_df['Open'],
            high=chart_df['High'],
            low=chart_df['Low'],
            close=chart_df['Close'],
            name='가격'
        ))
        
        if 'MA40' in chart_df.columns:
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['MA40'], 
                                     mode='lines', line=dict(color='orange', width=1.5), name='MA40'))
        if 'MA200' in chart_df.columns:
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['MA200'],
                                     mode='lines', line=dict(color='purple', width=1.5), name='MA200'))
        
        # 액션 마커 추가
        for action in chart_actions:
            if action['date'] >= signal_cutoff:
                if '매수' in action['action'] and '물타기' not in action['action']:
                    fig.add_trace(go.Scatter(
                        x=[action['date']], y=[action['price']],
                        mode='markers',
                        marker=dict(color='limegreen', size=14, symbol='triangle-up',
                                    line=dict(color='darkgreen', width=2)),
                        showlegend=False,
                        hovertemplate=f"🟢 매수<br>${action['price']:.2f}<br>{action['date'].strftime('%Y-%m-%d')}<extra></extra>"
                    ))
                elif '물타기' in action['action']:
                    fig.add_trace(go.Scatter(
                        x=[action['date']], y=[action['price']],
                        mode='markers',
                        marker=dict(color='dodgerblue', size=10, symbol='triangle-up',
                                    line=dict(color='darkblue', width=1)),
                        showlegend=False,
                        hovertemplate=f"{action['action']}<br>${action['price']:.2f}<br>{action['date'].strftime('%Y-%m-%d')}<extra></extra>"
                    ))
                elif '익절' in action['action']:
                    fig.add_trace(go.Scatter(
                        x=[action['date']], y=[action['price']],
                        mode='markers',
                        marker=dict(color='gold', size=14, symbol='diamond',
                                    line=dict(color='darkorange', width=2)),
                        showlegend=False,
                        hovertemplate=f"💰 익절<br>${action['price']:.2f}<br>+{action.get('return', 0):.1f}%<br>{action['date'].strftime('%Y-%m-%d')}<extra></extra>"
                    ))
        
        # 범례 (더미)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
            marker=dict(color='limegreen', size=12, symbol='triangle-up'), name='🟢 매수'))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
            marker=dict(color='dodgerblue', size=10, symbol='triangle-up'), name='🔵 물타기'))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
            marker=dict(color='gold', size=12, symbol='diamond'), name='💰 익절'))
        
        fig.update_layout(height=500, xaxis_rangeslider_visible=False, title=f"가격 차트 + 거래 액션 (최근 {lookback_days}일)")
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # ===== 시그널 내역 =====
        st.subheader(f"🔔 시그널 내역 (최근 {lookback_days}일)")
        
        filtered_buys = [bs for bs in buy_signals if bs['confirm_date'] >= signal_cutoff]
        filtered_sells = [ss for ss in sell_signals if ss['confirm_date'] >= signal_cutoff]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🟢 매수 시그널**")
            if filtered_buys:
                buy_df = pd.DataFrame([{
                    '매수일': bs['confirm_date'].strftime('%Y-%m-%d'),
                    '매수가': f"${bs['confirm_price']:.2f}",
                    'RSI': f"{bs['rsi_at_confirm']:.1f}"
                } for bs in sorted(filtered_buys, key=lambda x: x['confirm_date'], reverse=True)])
                st.dataframe(buy_df, use_container_width=True, hide_index=True)
            else:
                st.info("없음")
        
        with col2:
            st.markdown("**🔴 매도 시그널**")
            if filtered_sells:
                sell_df = pd.DataFrame([{
                    '매도일': ss['confirm_date'].strftime('%Y-%m-%d'),
                    '매도가': f"${ss['confirm_price']:.2f}"
                } for ss in sorted(filtered_sells, key=lambda x: x['confirm_date'], reverse=True)])
                st.dataframe(sell_df, use_container_width=True, hide_index=True)
            else:
                st.info("없음")
        
        st.divider()
        
        # ===== 전략 성과 (기간 내) =====
        filtered_trades = [t for t in trades if t['exit_date'] >= signal_cutoff]
        
        st.subheader(f"💹 전략 성과 (최근 {lookback_days}일) - 실제 금액 기준")
        st.caption(f"각 매수마다 동일 금액(${CAPITAL_PER_ENTRY:,}) 투자 가정")
        
        if filtered_trades:
            total_trades_period = len(filtered_trades)
            wins_period = len([t for t in filtered_trades if t['return'] > 0])
            total_invested_period = sum(t['num_buys'] * CAPITAL_PER_ENTRY for t in filtered_trades)
            total_profit_period = sum(t['num_buys'] * CAPITAL_PER_ENTRY * t['return'] / 100 for t in filtered_trades)
            total_return_period = (total_profit_period / total_invested_period * 100) if total_invested_period > 0 else 0
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("총 거래", f"{total_trades_period}회")
            with col2:
                st.metric("승률", f"{wins_period/total_trades_period*100:.0f}%")
            with col3:
                st.metric("총 투자금", f"${total_invested_period:,}")
            with col4:
                color = "🟢" if total_profit_period >= 0 else "🔴"
                st.metric("총 손익", f"{color} ${total_profit_period:+,.0f}")
            with col5:
                st.metric("금액 수익률", f"{total_return_period:+.1f}%")
            
            # 거래 내역
            st.markdown("**📋 거래 내역**")
            trade_df_period = pd.DataFrame([{
                '기간': f"{t['entry_dates'][0].strftime('%Y-%m-%d')} ~ {t['exit_date'].strftime('%Y-%m-%d')}",
                '물타기': f"{t['num_buys']}회",
                '투자금': f"${t['num_buys'] * CAPITAL_PER_ENTRY:,}",
                '평단가': f"${t['avg_price']:.2f}",
                '매도가': f"${t['exit_price']:.2f}",
                '손익': f"${t['num_buys'] * CAPITAL_PER_ENTRY * t['return'] / 100:+,.0f}",
                '수익률': f"{t['return']:+.1f}%"
            } for t in sorted(filtered_trades, key=lambda x: x['exit_date'], reverse=True)])
            st.dataframe(trade_df_period, use_container_width=True, hide_index=True)
        else:
            st.info(f"최근 {lookback_days}일간 완료된 거래 없음")
    
    # ===== 탭 2: 통합 뷰 =====
    with tab2:
        st.header("📈 통합 뷰 - 모든 거래 액션")
        
        all_actions = []
        for trade in trades:
            all_actions.append({
                'date': trade['entry_dates'][0],
                'action': '🟢 매수',
                'price': trade['entry_prices'][0],
                'position': 'LONG',
                'invested': CAPITAL_PER_ENTRY,
                'return': None
            })
            for i in range(1, trade['num_buys']):
                all_actions.append({
                    'date': trade['entry_dates'][i],
                    'action': f'🔵 물타기 ({i+1}회)',
                    'price': trade['entry_prices'][i],
                    'position': 'LONG (보유중)',
                    'invested': CAPITAL_PER_ENTRY * (i + 1),
                    'return': None
                })
            profit = trade['num_buys'] * CAPITAL_PER_ENTRY * trade['return'] / 100
            all_actions.append({
                'date': trade['exit_date'],
                'action': '💰 익절',
                'price': trade['exit_price'],
                'position': 'CLOSE',
                'invested': trade['num_buys'] * CAPITAL_PER_ENTRY,
                'return': trade['return'],
                'profit': profit
            })
        
        for p in positions:
            idx = positions.index(p)
            if idx == 0:
                all_actions.append({
                    'date': p['date'],
                    'action': '🟢 매수',
                    'price': p['price'],
                    'position': 'LONG (보유중)',
                    'invested': CAPITAL_PER_ENTRY,
                    'return': None
                })
            else:
                all_actions.append({
                    'date': p['date'],
                    'action': f'🔵 물타기 ({idx+1}회)',
                    'price': p['price'],
                    'position': 'LONG (보유중)',
                    'invested': CAPITAL_PER_ENTRY * (idx + 1),
                    'return': None
                })
        
        all_actions.sort(key=lambda x: x['date'], reverse=True)
        
        if all_actions:
            st.subheader("📋 액션 타임라인")
            action_df = pd.DataFrame([{
                '날짜': a['date'].strftime('%Y-%m-%d'),
                '액션': a['action'],
                '가격': f"${a['price']:.2f}",
                '포지션': a['position'],
                '투자금': f"${a['invested']:,}",
                '손익': f"${a.get('profit', 0):+,.0f} ({a['return']:+.1f}%)" if a['return'] else '-'
            } for a in all_actions[:50]])
            st.dataframe(action_df, use_container_width=True, hide_index=True)
        else:
            st.info("거래 내역이 없습니다.")
    
    # ===== 탭 3: 전체 성과 =====
    with tab3:
        st.header("📋 전체 성과")
        
        if trades:
            total_trades = len(trades)
            wins = len([t for t in trades if t['return'] > 0])
            total_invested = sum(t['num_buys'] * CAPITAL_PER_ENTRY for t in trades)
            total_profit = sum(t['num_buys'] * CAPITAL_PER_ENTRY * t['return'] / 100 for t in trades)
            total_return = (total_profit / total_invested * 100) if total_invested > 0 else 0
            max_water = max(t['num_buys'] for t in trades)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("총 거래", f"{total_trades}회")
            with col2:
                st.metric("승률", f"{wins/total_trades*100:.0f}%")
            with col3:
                st.metric("총 투자금", f"${total_invested:,}")
            with col4:
                st.metric("총 손익", f"${total_profit:+,.0f}")
            with col5:
                st.metric("금액 수익률", f"{total_return:+.1f}%")
            
            st.metric("최대 물타기", f"{max_water}회 (최대 ${max_water * CAPITAL_PER_ENTRY:,} 필요)")
            
            st.divider()
            st.subheader("📋 거래 내역")
            trade_df = pd.DataFrame([{
                '기간': f"{t['entry_dates'][0].strftime('%Y-%m-%d')} ~ {t['exit_date'].strftime('%Y-%m-%d')}",
                '물타기': f"{t['num_buys']}회",
                '투자금': f"${t['num_buys'] * CAPITAL_PER_ENTRY:,}",
                '평단가': f"${t['avg_price']:.2f}",
                '매도가': f"${t['exit_price']:.2f}",
                '손익': f"${t['num_buys'] * CAPITAL_PER_ENTRY * t['return'] / 100:+,.0f}",
                '수익률': f"{t['return']:+.1f}%"
            } for t in sorted(trades, key=lambda x: x['exit_date'], reverse=True)])
            st.dataframe(trade_df, use_container_width=True, hide_index=True)
        else:
            st.info("완료된 거래가 없습니다.")


if __name__ == "__main__":
    main()
