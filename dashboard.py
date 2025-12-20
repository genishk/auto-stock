"""
Auto-Stock 대시보드 v3
streamlit run dashboard.py

탭 구성:
1. 📊 현재 상태 - 가격, 최근 신호, 포지션
2. 🏆 패턴 순위 - 복합 점수 기반 순위
3. ✅ 검증된 패턴 - 14개 패턴 상세
4. 📈 패턴 분석 - 발생 빈도, Train/Test 비교
5. 📑 검증 과정 - 2단계 검증 결과
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import json
from datetime import datetime
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.cache import DataCache
from src.data.fetcher import DataFetcher
from src.data.validator import DataValidator
from src.features.technical import TechnicalIndicators
from src.discovery.validated_patterns import (
    VALIDATED_PATTERNS, 
    get_validated_patterns, 
    check_signals
)
from src.utils.helpers import load_config

# 페이지 설정
st.set_page_config(
    page_title="Auto-Stock 패턴 분석",
    page_icon="📈",
    layout="wide"
)


@st.cache_data(ttl=3600)
def load_data(ticker: str):
    """데이터 로드 및 지표 계산"""
    config = load_config()
    cache = DataCache(
        cache_dir=str(project_root / config['data']['cache']['directory']),
        max_age_hours=24
    )
    
    df = cache.get(ticker)
    if df is None:
        fetcher = DataFetcher([ticker])
        data = fetcher.fetch(config['data']['period'])
        if ticker in data:
            df = data[ticker]
            df, _ = DataValidator.validate(df, ticker)
            cache.set(ticker, df)
    
    if df is not None:
        indicators = TechnicalIndicators(config.get('indicators', {}))
        df = indicators.calculate_all(df)
    
    return df


def load_position(ticker: str):
    """포지션 로드"""
    positions_path = project_root / "data" / "positions.json"
    if positions_path.exists():
        with open(positions_path, 'r') as f:
            positions = json.load(f)
        return positions.get(ticker)
    return None


def load_signal_history(ticker: str):
    """신호 히스토리 로드"""
    path = project_root / "data" / "signals" / f"{ticker}_signals.json"
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return None


def plot_price_with_signals(df: pd.DataFrame, signals: list):
    """가격 차트에 신호 표시"""
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.05
    )
    
    # 캔들스틱
    fig.add_trace(
        go.Candlestick(
            x=df.index[-120:],
            open=df['Open'].iloc[-120:],
            high=df['High'].iloc[-120:],
            low=df['Low'].iloc[-120:],
            close=df['Close'].iloc[-120:],
            name='가격'
        ),
        row=1, col=1
    )
    
    # 이동평균
    if 'ma_short' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index[-120:],
                y=df['ma_short'].iloc[-120:],
                name='MA20',
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
    
    if 'ma_medium' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index[-120:],
                y=df['ma_medium'].iloc[-120:],
                name='MA50',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
    
    # 신호 표시
    recent_dates = df.index[-120:]
    for sig in signals:
        sig_date = pd.Timestamp(sig['date'])
        if sig_date in recent_dates:
            idx = df.index.get_loc(sig_date)
            fig.add_annotation(
                x=sig_date,
                y=df['Low'].iloc[idx] * 0.98,
                text="🟢",
                showarrow=False,
                font=dict(size=16),
                row=1, col=1
            )
    
    # RSI
    if 'rsi' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index[-120:],
                y=df['rsi'].iloc[-120:],
                name='RSI',
                line=dict(color='purple', width=1)
            ),
            row=2, col=1
        )
        
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    
    fig.update_layout(
        height=600,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        title="가격 차트 (최근 120일)"
    )
    
    return fig


def plot_pattern_performance():
    """패턴 성과 비교 차트"""
    patterns = get_validated_patterns()
    
    data = []
    for p in patterns:
        data.append({
            'name': p.name,
            'category': p.category,
            'Train 승률': p.train_win_rate * 100,
            'Test 승률': p.test_win_rate * 100,
            'Train 수익': p.train_avg_return,
            'Test 수익': p.test_avg_return,
            'Lift': p.lift
        })
    
    df_perf = pd.DataFrame(data)
    df_perf = df_perf.sort_values('Test 승률', ascending=True)
    
    fig = go.Figure()
    
    # Train 승률
    fig.add_trace(go.Bar(
        name='Train 승률',
        y=df_perf['name'],
        x=df_perf['Train 승률'],
        orientation='h',
        marker_color='rgba(100, 149, 237, 0.6)'
    ))
    
    # Test 승률
    fig.add_trace(go.Bar(
        name='Test 승률',
        y=df_perf['name'],
        x=df_perf['Test 승률'],
        orientation='h',
        marker_color='rgba(50, 205, 50, 0.8)'
    ))
    
    # 기준선 (Test 랜덤 확률 약 35%)
    fig.add_vline(x=35.3, line_dash="dash", line_color="red",
                  annotation_text="기준선 (35.3%)")
    
    fig.update_layout(
        title="패턴별 승률 비교 (Train vs Test)",
        height=600,
        barmode='group',
        xaxis_title="승률 (%)",
        yaxis_title="패턴"
    )
    
    return fig


def find_buy_signals(df: pd.DataFrame, pattern, rsi_exit_threshold: float = 40.0):
    """
    실제 매수 시그널 찾기 (RSI 탈출 방식)
    
    조건: 시그널 구간이 끝나고 RSI가 threshold 이상으로 올라올 때
    → 그 시그널 구간의 마지막 시그널 날짜를 "매수 시그널"로 반환
    """
    buy_signals = []
    
    in_signal_zone = False
    last_signal_idx = None
    last_signal_date = None
    last_signal_price = None
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        is_signal = pattern.check(row)
        rsi = row.get('rsi', 50)
        
        if is_signal:
            # 시그널 구간 진입 또는 유지
            in_signal_zone = True
            last_signal_idx = idx
            last_signal_date = df.index[idx]
            last_signal_price = row['Close']
        else:
            # 시그널 없음
            if in_signal_zone:
                # 시그널 구간에서 나옴
                # RSI가 threshold 이상이면 → 매수 시그널 확정
                if rsi >= rsi_exit_threshold and last_signal_date is not None:
                    buy_signals.append({
                        'signal_date': last_signal_date,  # 마지막 시그널 날짜
                        'signal_price': last_signal_price,
                        'confirm_date': df.index[idx],    # RSI 탈출 확인 날짜
                        'confirm_price': row['Close'],
                        'rsi_at_confirm': rsi
                    })
                    in_signal_zone = False
                    last_signal_date = None
                # RSI가 아직 threshold 미만이면 대기 (다시 시그널 올 수도 있음)
    
    return buy_signals


def plot_pattern_occurrences(df: pd.DataFrame, pattern_name: str, rsi_threshold: float = 40.0):
    """특정 패턴의 발생 시점 시각화 (매수 시그널 포함)"""
    pattern = None
    for p in VALIDATED_PATTERNS:
        if p.name == pattern_name:
            pattern = p
            break
    
    if not pattern:
        return go.Figure(), []
    
    # Train/Test 분할 (70/30)
    split_idx = int(len(df) * 0.7)
    split_date = df.index[split_idx]
    
    # 모든 패턴 발생일 찾기
    all_signals = []
    for idx in range(len(df)):
        if pattern.check(df.iloc[idx]):
            all_signals.append({
                'date': df.index[idx],
                'idx': idx,
                'price': df['Close'].iloc[idx],
                'rsi': df['rsi'].iloc[idx],
                'period': 'Train' if idx < split_idx else 'Test'
            })
    
    # 실제 매수 시그널 찾기 (RSI 탈출 방식)
    buy_signals = find_buy_signals(df, pattern, rsi_threshold)
    buy_signal_dates = set(bs['signal_date'] for bs in buy_signals)
    
    fig = go.Figure()
    
    # 가격 차트
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        name='가격',
        line=dict(color='gray', width=1)
    ))
    
    # 일반 시그널 (매수 시그널 제외) - 연한 파란색
    normal_signals = [s for s in all_signals if s['date'] not in buy_signal_dates]
    normal_dates = [s['date'] for s in normal_signals]
    normal_prices = [s['price'] for s in normal_signals]
    
    fig.add_trace(go.Scatter(
        x=normal_dates,
        y=normal_prices,
        mode='markers',
        name=f'시그널 ({len(normal_signals)}회)',
        marker=dict(color='lightblue', size=8, symbol='circle', 
                    line=dict(color='blue', width=1)),
        hovertemplate='%{x}<br>가격: $%{y:.2f}<extra>시그널</extra>'
    ))
    
    # 실제 매수 시그널 (RSI 탈출 확인된 것) - 진한 초록색
    buy_dates = [bs['signal_date'] for bs in buy_signals]
    buy_prices = [bs['signal_price'] for bs in buy_signals]
    
    fig.add_trace(go.Scatter(
        x=buy_dates,
        y=buy_prices,
        mode='markers',
        name=f'★ 매수 시그널 ({len(buy_signals)}회)',
        marker=dict(color='limegreen', size=10, symbol='circle',
                    line=dict(color='darkgreen', width=2)),
        hovertemplate='%{x}<br>가격: $%{y:.2f}<br>★ 매수 시그널<extra></extra>'
    ))
    
    # Train/Test 분할선 - 문자열로 변환하여 호환성 문제 해결
    fig.add_shape(
        type="line",
        x0=str(split_date.date()),
        x1=str(split_date.date()),
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="red", width=2, dash="dash")
    )
    fig.add_annotation(
        x=str(split_date.date()),
        y=1.05,
        yref="paper",
        text="Train/Test 분할",
        showarrow=False,
        font=dict(color="red", size=10)
    )
    
    fig.update_layout(
        title=f"{pattern_name} 발생 시점 (★ = RSI {rsi_threshold}+ 탈출 후 매수)",
        height=500,
        xaxis_title="날짜",
        yaxis_title="가격"
    )
    
    return fig, buy_signals, all_signals


def show_pattern_details(pattern):
    """패턴 상세 정보 표시"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📚 Train 성과**")
        st.metric("발생 횟수", f"{pattern.train_occurrences}회")
        st.metric("승률", f"{pattern.train_win_rate*100:.1f}%")
        st.metric("평균 수익률", f"{pattern.train_avg_return:.1f}%")
    
    with col2:
        st.markdown("**🧪 Test 성과**")
        st.metric("발생 횟수", f"{pattern.test_occurrences}회")
        st.metric("승률", f"{pattern.test_win_rate*100:.1f}%", 
                  delta=f"기준선 대비 +{(pattern.test_win_rate*100 - 35.3):.1f}%p")
        st.metric("평균 수익률", f"{pattern.test_avg_return:.1f}%")
    
    st.metric("Lift (Test 승률 / 기준선)", f"{pattern.lift:.2f}x")
    
    # 조건 표시
    st.markdown("**📋 조건**")
    for indicator, (min_val, max_val) in pattern.conditions.items():
        if min_val <= -100:
            st.write(f"  - `{indicator}` < {max_val}")
        elif max_val >= 100:
            st.write(f"  - `{indicator}` > {min_val}")
        else:
            st.write(f"  - {min_val} ≤ `{indicator}` ≤ {max_val}")


def main():
    st.title("📈 Auto-Stock 패턴 분석 시스템")
    st.caption(f"마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # 사이드바
    st.sidebar.header("⚙️ 설정")
    
    config = load_config()
    ticker = st.sidebar.selectbox("종목", config['tickers'], index=0)
    # 전체 데이터 기간 계산 (로드 후 설정)
    lookback_days = st.sidebar.slider("신호 확인 기간 (일)", 30, 3650, 365)
    
    # 데이터 로드
    df = load_data(ticker)
    
    if df is None:
        st.error(f"❌ {ticker} 데이터를 불러올 수 없습니다.")
        return
    
    st.sidebar.success(f"✅ {len(df)}일 데이터 로드")
    st.sidebar.info(f"📅 {df.index[0].date()} ~ {df.index[-1].date()}")
    
    # 탭 구성
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 현재 상태",
        "🏆 패턴 순위",
        "✅ 검증된 패턴 (14개)",
        "📈 패턴 분석",
        "📑 검증 과정",
        "🔍 데이터 확인"
    ])
    
    # ===== 탭 1: 현재 상태 (물타기 전략) =====
    with tab1:
        st.header(f"📊 {ticker} 현재 상태")
        
        # 현재 가격 정보
        current = df['Close'].iloc[-1]
        prev = df['Close'].iloc[-2]
        change = (current / prev - 1) * 100
        rsi_now = df['rsi'].iloc[-1]
        
        # ===== 물타기 전략 시뮬레이션 (전체 기간) =====
        # RSI_Oversold_35 패턴 찾기
        rsi_pattern = None
        for p in VALIDATED_PATTERNS:
            if p.name == "RSI_Oversold_35":
                rsi_pattern = p
                break
        
        # 매수/매도 시그널 계산
        home_buy_signals = find_buy_signals(df, rsi_pattern, rsi_exit_threshold=60.0) if rsi_pattern else []
        
        # 매도 시그널 찾기 (RSI > 70 -> RSI <= 50)
        home_sell_signals = []
        in_overbought = False
        last_ob_date = None
        last_ob_price = None
        
        for idx in range(len(df)):
            rsi = df['rsi'].iloc[idx]
            if rsi > 70:
                in_overbought = True
                last_ob_date = df.index[idx]
                last_ob_price = df['Close'].iloc[idx]
            else:
                if in_overbought and rsi <= 50 and last_ob_date is not None:
                    home_sell_signals.append({
                        'signal_date': last_ob_date,
                        'signal_price': last_ob_price,
                        'confirm_date': df.index[idx],
                        'confirm_price': df['Close'].iloc[idx]
                    })
                    in_overbought = False
                    last_ob_date = None
        
        # 물타기 시뮬레이션
        all_buy_dates = {bs['signal_date']: bs for bs in home_buy_signals}
        all_sell_dates = {ss['signal_date']: ss for ss in home_sell_signals}
        
        home_trades = []
        home_positions = []
        
        for idx in range(len(df)):
            current_date = df.index[idx]
            current_price = df['Close'].iloc[idx]
            
            if home_positions:
                total_cost = sum(p['price'] for p in home_positions)
                avg_price = total_cost / len(home_positions)
                current_return = (current_price / avg_price - 1) * 100
                
                exit_reason = None
                exit_price = current_price
                
                if current_date in all_sell_dates:
                    exit_reason = "RSI 매도"
                    exit_price = all_sell_dates[current_date]['signal_price']
                elif current_return <= -15:
                    exit_reason = "-15% 손절"
                
                if exit_reason:
                    final_return = (exit_price / avg_price - 1) * 100
                    home_trades.append({
                        'entry_dates': [p['date'] for p in home_positions],
                        'avg_price': avg_price,
                        'num_buys': len(home_positions),
                        'exit_date': current_date,
                        'exit_price': exit_price,
                        'return': final_return,
                        'exit_reason': exit_reason
                    })
                    home_positions = []
            
            if current_date in all_buy_dates:
                home_positions.append({
                    'date': current_date,
                    'price': all_buy_dates[current_date]['signal_price']
                })
        
        # ===== 현재 상태 표시 =====
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("현재가", f"${current:.2f}", f"{change:+.2f}%")
        with col2:
            rsi_status = "🔴 과매도" if rsi_now < 35 else ("🟢 과매수" if rsi_now > 70 else "⚪ 중립")
            st.metric("RSI", f"{rsi_now:.1f}", delta=rsi_status)
        with col3:
            if home_positions:
                avg_p = sum(p['price'] for p in home_positions) / len(home_positions)
                unrealized = (current / avg_p - 1) * 100
                st.metric("보유 상태", f"{len(home_positions)}회 물타기", delta=f"{unrealized:+.1f}%")
            else:
                st.metric("보유 상태", "대기 중")
        with col4:
            if home_trades:
                win_rate = len([t for t in home_trades if t['return'] > 0]) / len(home_trades) * 100
                st.metric("전체 승률", f"{win_rate:.0f}%")
        
        st.divider()
        
        # ===== 현재 포지션 상세 =====
        if home_positions:
            st.subheader("💰 현재 보유 포지션")
            avg_price = sum(p['price'] for p in home_positions) / len(home_positions)
            unrealized = (current / avg_price - 1) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("평균 매수가", f"${avg_price:.2f}")
            with col2:
                st.metric("물타기 횟수", f"{len(home_positions)}회")
            with col3:
                color = "🟢" if unrealized >= 0 else "🔴"
                st.metric("미실현 손익", f"{color} {unrealized:+.1f}%")
            
            # 매수 내역
            st.markdown("**📋 매수 내역**")
            pos_df = pd.DataFrame([{
                '매수일': p['date'].strftime('%Y-%m-%d'),
                '매수가': f"${p['price']:.2f}",
                '현재 손익': f"{(current/p['price']-1)*100:+.1f}%"
            } for p in home_positions])
            st.dataframe(pos_df, use_container_width=True, hide_index=True)
            
            # 매도 조건 안내
            st.info(f"""
            **📤 매도 조건:**
            - RSI > 70 발생 후 → RSI ≤ 50 탈출 시 매도
            - 평단가 대비 -15% 손절 (현재: {unrealized:+.1f}%)
            """)
        else:
            st.subheader("⏳ 대기 중")
            st.info("현재 보유 포지션이 없습니다. 매수 시그널 대기 중...")
        
        st.divider()
        
        # ===== 최근 시그널 알림 (슬라이더 기간) =====
        st.subheader(f"🔔 시그널 내역 (최근 {lookback_days}일)")
        
        # 슬라이더 기간 내 시그널 필터링
        signal_cutoff = df.index[-1] - pd.Timedelta(days=lookback_days)
        
        # 기간 내 매수/매도 시그널
        filtered_buys = [bs for bs in home_buy_signals if bs['signal_date'] >= signal_cutoff]
        filtered_sells = [ss for ss in home_sell_signals if ss['signal_date'] >= signal_cutoff]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🟢 매수 시그널**")
            if filtered_buys:
                buy_df = pd.DataFrame([{
                    '날짜': bs['signal_date'].strftime('%Y-%m-%d'),
                    '가격': f"${bs['signal_price']:.2f}"
                } for bs in sorted(filtered_buys, key=lambda x: x['signal_date'], reverse=True)])
                st.dataframe(buy_df, use_container_width=True, hide_index=True)
            else:
                st.info("없음")
        
        with col2:
            st.markdown("**🔴 매도 시그널**")
            if filtered_sells:
                sell_df = pd.DataFrame([{
                    '날짜': ss['signal_date'].strftime('%Y-%m-%d'),
                    '가격': f"${ss['signal_price']:.2f}"
                } for ss in sorted(filtered_sells, key=lambda x: x['signal_date'], reverse=True)])
                st.dataframe(sell_df, use_container_width=True, hide_index=True)
            else:
                st.info("없음")
        
        # RSI 상태 알림
        if rsi_now < 35:
            st.warning(f"⚠️ RSI가 35 미만입니다 ({rsi_now:.1f}). 매수 시그널 구간 진입!")
        elif rsi_now > 70:
            st.warning(f"⚠️ RSI가 70 초과입니다 ({rsi_now:.1f}). 매도 시그널 구간 진입!")
        
        st.divider()
        
        # ===== 가격 차트 =====
        st.subheader("📉 가격 차트")
        
        # 슬라이더 기간에 맞는 차트
        chart_df = df[df.index >= signal_cutoff]
        
        fig_home = go.Figure()
        
        # 캔들스틱
        fig_home.add_trace(go.Candlestick(
            x=chart_df.index,
            open=chart_df['Open'],
            high=chart_df['High'],
            low=chart_df['Low'],
            close=chart_df['Close'],
            name='가격'
        ))
        
        # 매수 시그널 표시
        for bs in filtered_buys:
            fig_home.add_trace(go.Scatter(
                x=[bs['signal_date']],
                y=[bs['signal_price']],
                mode='markers',
                marker=dict(color='limegreen', size=14, symbol='triangle-up',
                            line=dict(color='darkgreen', width=2)),
                showlegend=False,
                hovertemplate=f"매수: ${bs['signal_price']:.2f}<br>{bs['signal_date'].strftime('%Y-%m-%d')}<extra></extra>"
            ))
        
        # 매도 시그널 표시
        for ss in filtered_sells:
            fig_home.add_trace(go.Scatter(
                x=[ss['signal_date']],
                y=[ss['signal_price']],
                mode='markers',
                marker=dict(color='red', size=14, symbol='triangle-down',
                            line=dict(color='darkred', width=2)),
                showlegend=False,
                hovertemplate=f"매도: ${ss['signal_price']:.2f}<br>{ss['signal_date'].strftime('%Y-%m-%d')}<extra></extra>"
            ))
        
        fig_home.update_layout(
            height=500,
            xaxis_rangeslider_visible=False,
            title=f"가격 차트 (최근 {lookback_days}일)"
        )
        
        st.plotly_chart(fig_home, use_container_width=True)
        
        st.divider()
        
        # ===== 전략 성과 요약 (슬라이더 기간) =====
        # 기간 내 거래만 필터링
        filtered_trades = [t for t in home_trades if t['exit_date'] >= signal_cutoff]
        
        st.subheader(f"📈 전략 성과 (최근 {lookback_days}일)")
        
        if filtered_trades:
            total_trades = len(filtered_trades)
            wins = len([t for t in filtered_trades if t['return'] > 0])
            total_return = sum(t['return'] for t in filtered_trades)
            avg_return = total_return / total_trades
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("총 거래", f"{total_trades}회")
            with col2:
                st.metric("승률", f"{wins/total_trades*100:.0f}%")
            with col3:
                st.metric("평균 수익률", f"{avg_return:+.1f}%")
            with col4:
                st.metric("누적 수익률", f"{total_return:+.1f}%")
            
            # 거래 내역 (기간 내 전체)
            st.markdown("**📋 거래 내역**")
            sorted_trades = sorted(filtered_trades, key=lambda x: x['exit_date'], reverse=True)
            trade_df = pd.DataFrame([{
                '기간': f"{t['entry_dates'][0].strftime('%Y-%m-%d')} ~ {t['exit_date'].strftime('%Y-%m-%d')}",
                '물타기': f"{t['num_buys']}회",
                '평단가': f"${t['avg_price']:.2f}",
                '매도가': f"${t['exit_price']:.2f}",
                '수익률': f"{t['return']:+.1f}%",
                '사유': t['exit_reason']
            } for t in sorted_trades])
            st.dataframe(trade_df, use_container_width=True, hide_index=True)
        else:
            st.info(f"최근 {lookback_days}일간 완료된 거래 없음")
    
    # ===== 탭 2: 패턴 순위 =====
    with tab2:
        st.header("🏆 패턴 순위")
        
        st.markdown("""
        **순위 산정 공식:**
        ```
        기대 수익 = 승률 × 평균 수익률
        신뢰도 = Test 발생 / (Test 발생 + 20)
        최종 점수 = 기대 수익 × 신뢰도
        ```
        
        - **기대 수익**: 이 패턴에 베팅했을 때 평균적으로 기대할 수 있는 수익
        - **신뢰도**: 발생 횟수가 적으면 우연일 수 있으므로 패널티 적용
        """)
        
        # 순위 계산
        ranking_data = []
        for p in VALIDATED_PATTERNS:
            expected_return = p.test_win_rate * p.test_avg_return
            confidence = p.test_occurrences / (p.test_occurrences + 20)
            final_score = expected_return * confidence
            annual_occurrences = p.test_occurrences / 3  # Test 기간 약 3년
            
            ranking_data.append({
                'pattern': p,
                'expected_return': expected_return,
                'confidence': confidence,
                'final_score': final_score,
                'annual_occurrences': annual_occurrences
            })
        
        # 점수순 정렬
        ranking_data.sort(key=lambda x: x['final_score'], reverse=True)
        
        # 순위 테이블
        st.subheader("📊 종합 순위")
        
        ranking_table = []
        for i, item in enumerate(ranking_data):
            p = item['pattern']
            ranking_table.append({
                '순위': i + 1,
                '패턴': p.name,
                '카테고리': p.category,
                'Test 승률': f"{p.test_win_rate*100:.1f}%",
                '평균 수익률': f"{p.test_avg_return:.1f}%",
                '기대 수익': f"{item['expected_return']:.2f}%",
                'Test 발생': f"{p.test_occurrences}회",
                '연간 발생': f"~{item['annual_occurrences']:.0f}회",
                '신뢰도': f"{item['confidence']*100:.0f}%",
                '최종 점수': f"{item['final_score']:.2f}",
                'Lift': f"{p.lift:.2f}x"
            })
        
        ranking_df = pd.DataFrame(ranking_table)
        st.dataframe(ranking_df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # 상위 3개 패턴 상세
        st.subheader("🥇 상위 3개 패턴 상세")
        
        for i, item in enumerate(ranking_data[:3]):
            p = item['pattern']
            medal = ["🥇", "🥈", "🥉"][i]
            
            with st.expander(f"{medal} {i+1}위: {p.name} (점수: {item['final_score']:.2f})", expanded=(i==0)):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**📈 수익 지표**")
                    st.metric("Test 승률", f"{p.test_win_rate*100:.1f}%")
                    st.metric("평균 수익률", f"{p.test_avg_return:.1f}%")
                    st.metric("기대 수익", f"{item['expected_return']:.2f}%")
                
                with col2:
                    st.markdown("**📊 발생 통계**")
                    st.metric("Test 발생", f"{p.test_occurrences}회")
                    st.metric("연간 발생", f"~{item['annual_occurrences']:.0f}회")
                    st.metric("신뢰도", f"{item['confidence']*100:.0f}%")
                
                with col3:
                    st.markdown("**🔍 검증 결과**")
                    st.metric("Lift", f"{p.lift:.2f}x")
                    st.metric("Train 승률", f"{p.train_win_rate*100:.1f}%")
                    st.metric("Train 발생", f"{p.train_occurrences}회")
                
                st.markdown(f"**설명:** {p.description}")
                
                st.markdown("**조건:**")
                for indicator, (min_val, max_val) in p.conditions.items():
                    if min_val <= -100:
                        st.write(f"  - `{indicator}` < {max_val}")
                    elif max_val >= 100:
                        st.write(f"  - `{indicator}` > {min_val}")
                    else:
                        st.write(f"  - {min_val} ≤ `{indicator}` ≤ {max_val}")
        
        st.divider()
        
        # 시각화: 기대 수익 vs 신뢰도
        st.subheader("📉 기대 수익 vs 신뢰도 분포")
        
        fig = go.Figure()
        
        for item in ranking_data:
            p = item['pattern']
            fig.add_trace(go.Scatter(
                x=[item['confidence'] * 100],
                y=[item['expected_return']],
                mode='markers+text',
                name=p.name,
                text=[p.name.replace('_', ' ')],
                textposition='top center',
                marker=dict(size=item['final_score'] * 3 + 10),
                hovertemplate=(
                    f"<b>{p.name}</b><br>"
                    f"기대 수익: {item['expected_return']:.2f}%<br>"
                    f"신뢰도: {item['confidence']*100:.0f}%<br>"
                    f"최종 점수: {item['final_score']:.2f}<br>"
                    f"<extra></extra>"
                )
            ))
        
        fig.update_layout(
            title="패턴 분포 (원 크기 = 최종 점수)",
            xaxis_title="신뢰도 (%)",
            yaxis_title="기대 수익 (%)",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # ===== 탭 3: 검증된 패턴 =====
    with tab3:
        st.header("✅ 검증된 패턴 (14개)")
        
        st.markdown("""
        **검증 기준:**
        - 발생도 검증: 발견 기간과 검증 기간 모두 꾸준히 발생
        - 수익률 검증: Train/Test 모두 랜덤보다 5%p+ 높은 승률, Lift 1.2x 이상
        """)
        
        # 성과 비교 차트
        fig = plot_pattern_performance()
        st.plotly_chart(fig, use_container_width=True)
        
        # 패턴 상세 리스트
        st.subheader("📋 패턴 상세")
        
        # 카테고리별 그룹화
        categories = {}
        for p in VALIDATED_PATTERNS:
            if p.category not in categories:
                categories[p.category] = []
            categories[p.category].append(p)
        
        for category, patterns in categories.items():
            st.markdown(f"### {category} ({len(patterns)}개)")
            
            for pattern in sorted(patterns, key=lambda x: x.test_win_rate, reverse=True):
                with st.expander(f"📌 {pattern.name} - Test 승률: {pattern.test_win_rate*100:.0f}%"):
                    st.markdown(f"**설명:** {pattern.description}")
                    show_pattern_details(pattern)
    
    # ===== 탭 4: 패턴 분석 =====
    with tab4:
        st.header("📈 패턴 발생 분석")
        
        # 패턴 선택
        pattern_names = [p.name for p in VALIDATED_PATTERNS]
        default_idx = pattern_names.index("RSI_Oversold_35") if "RSI_Oversold_35" in pattern_names else 0
        selected_pattern = st.selectbox("패턴 선택", pattern_names, index=default_idx)
        
        # RSI 탈출 기준 슬라이더
        st.markdown("**매수 시그널 조건**: 시그널 종료 후 RSI가 아래 값 이상이면 매수")
        rsi_threshold = st.slider("RSI 탈출 기준 (매수)", 35, 70, 60, 
                                   help="시그널 구간 후 RSI가 이 값 이상이면 '매수 시그널'로 확정")
        
        # 발생 시점 차트
        fig, buy_signals, all_signals = plot_pattern_occurrences(df, selected_pattern, rsi_threshold)
        st.plotly_chart(fig, use_container_width=True)
        
        # 매수 시그널 통계
        st.subheader("📊 시그널 통계")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("전체 시그널", f"{len(all_signals)}회")
        with col2:
            st.metric("★ 매수 시그널", f"{len(buy_signals)}회", 
                      delta=f"RSI {rsi_threshold}+ 탈출 확인")
        with col3:
            reduction = (1 - len(buy_signals) / len(all_signals)) * 100 if all_signals else 0
            st.metric("필터링 비율", f"{reduction:.0f}% 감소")
        
        st.divider()
        
        # 최근 매수 시그널 리스트
        if buy_signals:
            st.subheader("★ 최근 매수 시그널")
            recent_buys = sorted(buy_signals, key=lambda x: x['signal_date'], reverse=True)[:10]
            
            buy_df = pd.DataFrame([{
                '시그널 날짜': bs['signal_date'].strftime('%Y-%m-%d'),
                '시그널 가격': f"${bs['signal_price']:.2f}",
                '확인 날짜': bs['confirm_date'].strftime('%Y-%m-%d'),
                '확인 가격': f"${bs['confirm_price']:.2f}",
                'RSI (확인 시)': f"{bs['rsi_at_confirm']:.1f}"
            } for bs in recent_buys])
            
            st.dataframe(buy_df, use_container_width=True, hide_index=True)
        else:
            st.info("매수 시그널 없음")
        
        st.divider()
        
        # 선택된 패턴 정보
        st.subheader("📋 패턴 정보")
        for p in VALIDATED_PATTERNS:
            if p.name == selected_pattern:
                st.markdown(f"**{p.name}**: {p.description}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Train 발생", f"{p.train_occurrences}회")
                    st.metric("Train 승률", f"{p.train_win_rate*100:.1f}%")
                with col2:
                    st.metric("Test 발생", f"{p.test_occurrences}회")
                    st.metric("Test 승률", f"{p.test_win_rate*100:.1f}%")
                with col3:
                    st.metric("Lift", f"{p.lift:.2f}x")
                    st.metric("Test 평균 수익", f"{p.test_avg_return:.1f}%")
                break
        
        st.divider()
        
        # ===== 매도 시그널 분석 섹션 =====
        st.subheader("📤 매도 시그널 분석 (RSI 과매수)")
        st.caption("조건: RSI > 70 (과매수) 시그널 발생 후 → RSI ≤ X (탈출) 시 매도")
        
        # RSI 과매수 탈출 기준 슬라이더
        sell_rsi_threshold = st.slider("RSI 탈출 기준 (매도)", 30, 70, 50, 
                                        help="과매수 구간 후 RSI가 이 값 이하이면 '매도 시그널'로 확정")
        
        # RSI 과매수 시그널 찾기 (RSI > 70)
        overbought_signals = []
        for idx in range(len(df)):
            if df['rsi'].iloc[idx] > 70:
                overbought_signals.append({
                    'date': df.index[idx],
                    'idx': idx,
                    'price': df['Close'].iloc[idx],
                    'rsi': df['rsi'].iloc[idx]
                })
        
        # 매도 시그널 찾기 (과매수 → 탈출)
        sell_signals = []
        in_overbought = False
        last_overbought_idx = None
        last_overbought_date = None
        last_overbought_price = None
        
        for idx in range(len(df)):
            rsi = df['rsi'].iloc[idx]
            
            if rsi > 70:
                in_overbought = True
                last_overbought_idx = idx
                last_overbought_date = df.index[idx]
                last_overbought_price = df['Close'].iloc[idx]
            else:
                if in_overbought and rsi <= sell_rsi_threshold and last_overbought_date is not None:
                    sell_signals.append({
                        'signal_date': last_overbought_date,
                        'signal_price': last_overbought_price,
                        'confirm_date': df.index[idx],
                        'confirm_price': df['Close'].iloc[idx],
                        'rsi_at_confirm': rsi
                    })
                    in_overbought = False
                    last_overbought_date = None
        
        # 매도 시그널 차트
        fig_sell = go.Figure()
        
        # 가격 차트
        fig_sell.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            name='가격',
            line=dict(color='gray', width=1)
        ))
        
        # 과매수 시그널 (연한 빨간색)
        sell_signal_dates = set(ss['signal_date'] for ss in sell_signals)
        normal_overbought = [s for s in overbought_signals if s['date'] not in sell_signal_dates]
        
        fig_sell.add_trace(go.Scatter(
            x=[s['date'] for s in normal_overbought],
            y=[s['price'] for s in normal_overbought],
            mode='markers',
            name=f'과매수 시그널 ({len(normal_overbought)}회)',
            marker=dict(color='lightsalmon', size=8, symbol='circle',
                        line=dict(color='red', width=1)),
            hovertemplate='%{x}<br>가격: $%{y:.2f}<br>RSI > 70<extra></extra>'
        ))
        
        # 매도 시그널 (진한 빨간색)
        fig_sell.add_trace(go.Scatter(
            x=[ss['signal_date'] for ss in sell_signals],
            y=[ss['signal_price'] for ss in sell_signals],
            mode='markers',
            name=f'★ 매도 시그널 ({len(sell_signals)}회)',
            marker=dict(color='red', size=10, symbol='circle',
                        line=dict(color='darkred', width=2)),
            hovertemplate='%{x}<br>가격: $%{y:.2f}<br>★ 매도 시그널<extra></extra>'
        ))
        
        fig_sell.update_layout(
            title=f"RSI 과매수 시그널 (★ = RSI {sell_rsi_threshold} 이하 탈출 후 매도)",
            height=500,
            xaxis_title="날짜",
            yaxis_title="가격"
        )
        
        st.plotly_chart(fig_sell, use_container_width=True)
        
        # 매도 시그널 통계
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("과매수 시그널", f"{len(overbought_signals)}회")
        with col2:
            st.metric("★ 매도 시그널", f"{len(sell_signals)}회",
                      delta=f"RSI {sell_rsi_threshold} 이하 탈출")
        with col3:
            sell_reduction = (1 - len(sell_signals) / len(overbought_signals)) * 100 if overbought_signals else 0
            st.metric("필터링 비율", f"{sell_reduction:.0f}% 감소")
        
        # 최근 매도 시그널 리스트
        if sell_signals:
            st.markdown("**★ 최근 매도 시그널**")
            recent_sells = sorted(sell_signals, key=lambda x: x['signal_date'], reverse=True)[:10]
            
            sell_df = pd.DataFrame([{
                '시그널 날짜': ss['signal_date'].strftime('%Y-%m-%d'),
                '시그널 가격': f"${ss['signal_price']:.2f}",
                '확인 날짜': ss['confirm_date'].strftime('%Y-%m-%d'),
                '확인 가격': f"${ss['confirm_price']:.2f}",
                'RSI (확인 시)': f"{ss['rsi_at_confirm']:.1f}"
            } for ss in recent_sells])
            
            st.dataframe(sell_df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # ===== 매수 + 매도 통합 차트 =====
        st.subheader("🎯 매수/매도 시그널 통합 차트")
        st.caption(f"매수: RSI < 35 → RSI ≥ {rsi_threshold} 탈출 | 매도: RSI > 70 → RSI ≤ {sell_rsi_threshold} 탈출 | 손절: -10%")
        
        fig_combined = go.Figure()
        
        # 가격 차트
        fig_combined.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            name='가격',
            line=dict(color='gray', width=1.5)
        ))
        
        # 매수 시그널 (초록색)
        fig_combined.add_trace(go.Scatter(
            x=[bs['signal_date'] for bs in buy_signals],
            y=[bs['signal_price'] for bs in buy_signals],
            mode='markers',
            name=f'🟢 매수 ({len(buy_signals)}회)',
            marker=dict(color='limegreen', size=12, symbol='triangle-up',
                        line=dict(color='darkgreen', width=2)),
            hovertemplate='%{x}<br>매수: $%{y:.2f}<extra>🟢 매수 시그널</extra>'
        ))
        
        # 매도 시그널 (빨간색)
        fig_combined.add_trace(go.Scatter(
            x=[ss['signal_date'] for ss in sell_signals],
            y=[ss['signal_price'] for ss in sell_signals],
            mode='markers',
            name=f'🔴 매도 ({len(sell_signals)}회)',
            marker=dict(color='red', size=12, symbol='triangle-down',
                        line=dict(color='darkred', width=2)),
            hovertemplate='%{x}<br>매도: $%{y:.2f}<extra>🔴 매도 시그널</extra>'
        ))
        
        fig_combined.update_layout(
            title="매수/매도 시그널 통합",
            height=600,
            xaxis_title="날짜",
            yaxis_title="가격",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig_combined, use_container_width=True)
        
        # 통합 통계
        st.markdown("**📊 통합 통계**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🟢 매수 시그널", f"{len(buy_signals)}회")
        with col2:
            st.metric("🔴 매도 시그널", f"{len(sell_signals)}회")
        with col3:
            # 매수-매도 매칭 분석 (간단 버전)
            if buy_signals and sell_signals:
                st.metric("신호 비율", f"{len(sell_signals)/len(buy_signals):.1f}x")
            else:
                st.metric("신호 비율", "N/A")
        with col4:
            st.metric("손절 기준", "-15%")
        
        st.divider()
        
        # ===== 최종 전략 시뮬레이션 차트 (물타기) =====
        st.subheader("🎯 최종 전략: 물타기 시뮬레이션")
        st.markdown("""
        **전략:**
        - 매수 시그널 발생 시 → 추가 매수 (물타기, 평단가 낮춤)
        - 매도 조건 (먼저 발생하는 것):
          1. RSI 매도 시그널 (RSI > 70 → ≤50 탈출) → 전량 매도
          2. 평균 매수가 대비 -15% 손절 → 전량 매도
        """)
        
        # 물타기 전략 시뮬레이션
        all_buy_dates = {bs['signal_date']: bs for bs in buy_signals}
        all_sell_dates = {ss['signal_date']: ss for ss in sell_signals}
        
        trades = []
        positions = []  # 여러 포지션 보유 가능 (물타기)
        
        for idx in range(len(df)):
            current_date = df.index[idx]
            current_price = df['Close'].iloc[idx]
            
            # 포지션이 있을 때
            if positions:
                # 평균 매수가 계산
                total_cost = sum(p['price'] for p in positions)
                avg_price = total_cost / len(positions)
                current_return = (current_price / avg_price - 1) * 100
                
                exit_reason = None
                exit_price = current_price
                
                # 조건 1: RSI 매도 시그널
                if current_date in all_sell_dates:
                    exit_reason = "RSI 매도"
                    exit_price = all_sell_dates[current_date]['signal_price']
                
                # 조건 2: 손절 -15%
                elif current_return <= -15:
                    exit_reason = "-15% 손절"
                
                if exit_reason:
                    final_return = (exit_price / avg_price - 1) * 100
                    trades.append({
                        'entry_dates': [p['date'] for p in positions],
                        'entry_prices': [p['price'] for p in positions],
                        'avg_price': avg_price,
                        'num_buys': len(positions),
                        'exit_date': current_date,
                        'exit_price': exit_price,
                        'return': final_return,
                        'exit_reason': exit_reason
                    })
                    positions = []
            
            # 매수 시그널 시 추가 매수 (물타기)
            if current_date in all_buy_dates:
                positions.append({
                    'date': current_date,
                    'price': all_buy_dates[current_date]['signal_price']
                })
        
        # 최종 전략 차트
        fig_strategy = go.Figure()
        
        # 가격 차트
        fig_strategy.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            name='가격',
            line=dict(color='gray', width=1.5)
        ))
        
        # 각 거래 표시
        for trade in trades:
            # 모든 매수 포인트 표시
            for i, (buy_date, buy_price) in enumerate(zip(trade['entry_dates'], trade['entry_prices'])):
                # 첫 매수는 더 크게, 물타기는 작게
                size = 14 if i == 0 else 10
                fig_strategy.add_trace(go.Scatter(
                    x=[buy_date],
                    y=[buy_price],
                    mode='markers',
                    marker=dict(color='limegreen', size=size, symbol='triangle-up',
                                line=dict(color='darkgreen', width=2)),
                    showlegend=False,
                    hovertemplate=f"{'매수' if i == 0 else '물타기'}: ${buy_price:.2f}<br>{buy_date.strftime('%Y-%m-%d')}<extra></extra>"
                ))
            
            # 평균 매수가 라인 (첫 매수 ~ 매도 구간)
            if trade['num_buys'] > 1:
                fig_strategy.add_trace(go.Scatter(
                    x=[trade['entry_dates'][0], trade['exit_date']],
                    y=[trade['avg_price'], trade['avg_price']],
                    mode='lines',
                    line=dict(color='orange', width=1, dash='dash'),
                    showlegend=False,
                    hovertemplate=f"평단: ${trade['avg_price']:.2f}<extra></extra>"
                ))
            
            # 매도 포인트 (수익/손실에 따라 색상)
            sell_color = 'red' if trade['return'] < 0 else 'blue'
            
            fig_strategy.add_trace(go.Scatter(
                x=[trade['exit_date']],
                y=[trade['exit_price']],
                mode='markers',
                marker=dict(color=sell_color, size=14, symbol='triangle-down',
                            line=dict(color='darkred' if trade['return'] < 0 else 'darkblue', width=2)),
                showlegend=False,
                hovertemplate=f"매도: ${trade['exit_price']:.2f}<br>{trade['exit_date'].strftime('%Y-%m-%d')}<br>{trade['exit_reason']}<br>평단: ${trade['avg_price']:.2f}<br>매수 {trade['num_buys']}회<br>수익률: {trade['return']:+.1f}%<extra></extra>"
            ))
            
            # 거래 연결선 (평단 → 매도가)
            line_color = 'rgba(0,200,0,0.3)' if trade['return'] >= 0 else 'rgba(255,0,0,0.3)'
            fig_strategy.add_trace(go.Scatter(
                x=[trade['entry_dates'][-1], trade['exit_date']],
                y=[trade['avg_price'], trade['exit_price']],
                mode='lines',
                line=dict(color=line_color, width=2, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # 현재 보유 중인 포지션 표시
        if positions:
            avg_price = sum(p['price'] for p in positions) / len(positions)
            for i, p in enumerate(positions):
                size = 16 if i == 0 else 12
                fig_strategy.add_trace(go.Scatter(
                    x=[p['date']],
                    y=[p['price']],
                    mode='markers',
                    marker=dict(color='yellow', size=size, symbol='star',
                                line=dict(color='orange', width=2)),
                    showlegend=False,
                    hovertemplate=f"보유 중<br>{'첫 매수' if i == 0 else '물타기'}: ${p['price']:.2f}<br>{p['date'].strftime('%Y-%m-%d')}<extra></extra>"
                ))
            # 평단 표시
            fig_strategy.add_annotation(
                x=positions[-1]['date'],
                y=avg_price,
                text=f"평단: ${avg_price:.2f} ({len(positions)}회)",
                showarrow=True,
                arrowhead=2,
                arrowcolor="orange",
                font=dict(color="orange")
            )
        
        # 범례 추가 (더미)
        fig_strategy.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
            marker=dict(color='limegreen', size=12, symbol='triangle-up'),
            name='🟢 매수/물타기'))
        fig_strategy.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
            marker=dict(color='blue', size=12, symbol='triangle-down'),
            name='🔵 익절'))
        fig_strategy.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
            marker=dict(color='red', size=12, symbol='triangle-down'),
            name='🔴 손절'))
        fig_strategy.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
            line=dict(color='orange', dash='dash'),
            name='📊 평단가'))
        
        fig_strategy.update_layout(
            title="최종 전략: 물타기 시뮬레이션",
            height=650,
            xaxis_title="날짜",
            yaxis_title="가격",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig_strategy, use_container_width=True)
        
        # 거래 결과 통계
        st.markdown("**📊 거래 결과**")
        
        if trades:
            total_trades = len(trades)
            wins = [t for t in trades if t['return'] > 0]
            losses = [t for t in trades if t['return'] <= 0]
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("총 거래", f"{total_trades}회")
            with col2:
                win_rate = len(wins) / total_trades * 100
                st.metric("승률", f"{win_rate:.0f}%")
            with col3:
                avg_return = sum(t['return'] for t in trades) / total_trades
                st.metric("평균 수익률", f"{avg_return:+.1f}%")
            with col4:
                total_return = sum(t['return'] for t in trades)
                st.metric("총 수익률", f"{total_return:+.1f}%")
            with col5:
                if positions:
                    avg_price = sum(p['price'] for p in positions) / len(positions)
                    current_unrealized = (df['Close'].iloc[-1] / avg_price - 1) * 100
                    st.metric("미실현 수익", f"{current_unrealized:+.1f}%")
                else:
                    st.metric("현재 상태", "대기 중")
            
            # 거래 상세 테이블
            st.markdown("**📋 거래 내역**")
            trade_df = pd.DataFrame([{
                '첫 매수일': t['entry_dates'][0].strftime('%Y-%m-%d'),
                '매수 횟수': f"{t['num_buys']}회",
                '평단가': f"${t['avg_price']:.2f}",
                '매도일': t['exit_date'].strftime('%Y-%m-%d'),
                '매도가': f"${t['exit_price']:.2f}",
                '수익률': f"{t['return']:+.1f}%",
                '매도 사유': t['exit_reason']
            } for t in sorted(trades, key=lambda x: x['entry_dates'][0], reverse=True)])
            
            st.dataframe(trade_df, use_container_width=True, hide_index=True)
            
            # 매도 사유별 통계
            st.markdown("**📈 매도 사유별 통계**")
            reason_stats = {}
            for t in trades:
                reason = t['exit_reason']
                if reason not in reason_stats:
                    reason_stats[reason] = {'count': 0, 'returns': []}
                reason_stats[reason]['count'] += 1
                reason_stats[reason]['returns'].append(t['return'])
            
            reason_df = pd.DataFrame([{
                '매도 사유': reason,
                '횟수': stats['count'],
                '평균 수익률': f"{sum(stats['returns'])/len(stats['returns']):+.1f}%"
            } for reason, stats in reason_stats.items()])
            
            st.dataframe(reason_df, use_container_width=True, hide_index=True)
        else:
            st.info("거래 내역이 없습니다")
        
        if positions:
            avg_price = sum(p['price'] for p in positions) / len(positions)
            current_unrealized = (df['Close'].iloc[-1] / avg_price - 1) * 100
            st.warning(f"⚠️ 현재 보유 중: {len(positions)}회 물타기, 평단 ${avg_price:.2f}, 미실현 {current_unrealized:+.1f}%")
    
    # ===== 탭 5: 검증 과정 =====
    with tab5:
        st.header("📑 2단계 검증 과정")
        
        st.markdown("""
        ## 검증 과정 요약
        
        ### 1단계: 패턴 발견 + 발생도 검증
        
        1. **수익 포인트 정의**: 60일 후 10% 이상 수익인 날 → 600개 케이스
        2. **패턴 발견**: 앞 402개 수익 케이스 직전의 공통 특징 분석
        3. **패턴 정의**: 33개 패턴 정의 (RSI, 모멘텀, 볼린저, 추세, 거래량, 복합)
        4. **발생도 검증**: 검증 기간에도 꾸준히 발생하는지 확인 → 28개 통과
        
        ### 2단계: 수익률 검증
        
        1. **Train/Test 분할**: 70/30 (2015-12 ~ 2022-12 / 2022-12 ~ 2025-12)
        2. **기준선 계산**: 
           - Train: 20.5% (랜덤 확률)
           - Test: 35.3% (랜덤 확률)
        3. **검증 기준**:
           - 최소 발생 횟수 (Train 20회+, Test 10회+)
           - 승률 > 랜덤 + 5%p
           - Lift > 1.2x (랜덤보다 20% 이상 좋음)
        4. **결과**: 14개 통과
        """)
        
        st.divider()
        
        # 검증 결과 테이블
        st.subheader("📊 최종 검증 결과")
        
        results = []
        for p in VALIDATED_PATTERNS:
            results.append({
                '패턴': p.name,
                '카테고리': p.category,
                'Train 발생': p.train_occurrences,
                'Train 승률': f"{p.train_win_rate*100:.1f}%",
                'Test 발생': p.test_occurrences,
                'Test 승률': f"{p.test_win_rate*100:.1f}%",
                'Lift': f"{p.lift:.2f}x",
                'Test 평균 수익': f"{p.test_avg_return:.1f}%"
            })
        
        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values('Test 승률', ascending=False)
        
        st.dataframe(result_df, use_container_width=True)
        
        st.divider()
        
        st.markdown("""
        ## 검증 논리 확인
        
        | 질문 | 답변 |
        |------|------|
        | 패턴 정의와 Test 기간이 분리됐나? | ✅ 패턴은 앞 402개 수익케이스에서, Test는 2022-12 이후 |
        | Train에서 과적합됐나? | ❌ Train 승률 20-40% (랜덤과 비슷) |
        | Test에서 효과가 있나? | ✅ Test 승률 50-100% (랜덤 35% 대비 우수) |
        | 통계적으로 유의미한가? | ✅ Test에서 10-60회 발생 |
        """)
    
    # ===== 탭 6: 데이터 확인 =====
    with tab6:
        st.header("🔍 데이터 확인")
        
        # 캐시 정보
        cache_dir = project_root / "data" / "cache"
        metadata_file = cache_dir / "metadata.json"
        
        if metadata_file.exists():
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            if ticker in metadata:
                cache_info = metadata[ticker]
                cached_at = cache_info.get('cached_at', 'N/A')
                
                st.success(f"✅ 데이터 캐시 정상")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("캐시 저장 시간", cached_at[:19].replace('T', ' '))
                with col2:
                    st.metric("총 거래일", f"{cache_info.get('rows', 'N/A')}일")
                with col3:
                    st.metric("데이터 기간", f"{cache_info.get('start_date', '')} ~ {cache_info.get('end_date', '')}")
        else:
            st.warning("캐시 메타데이터 없음")
        
        st.divider()
        
        # 최근 데이터 테이블
        st.subheader(f"📊 최근 데이터 (마지막 30일)")
        
        recent_df = df.tail(30).copy()
        recent_df = recent_df.sort_index(ascending=False)
        
        display_df = pd.DataFrame({
            '날짜': recent_df.index.strftime('%Y-%m-%d'),
            '시가': recent_df['Open'].apply(lambda x: f"${x:.2f}"),
            '고가': recent_df['High'].apply(lambda x: f"${x:.2f}"),
            '저가': recent_df['Low'].apply(lambda x: f"${x:.2f}"),
            '종가': recent_df['Close'].apply(lambda x: f"${x:.2f}"),
            '거래량': recent_df['Volume'].apply(lambda x: f"{x/1e6:.1f}M"),
            'RSI': recent_df['rsi'].apply(lambda x: f"{x:.1f}"),
        })
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # 데이터 무결성 확인
        st.subheader("🔒 데이터 무결성")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            last_date = df.index[-1]
            today = pd.Timestamp.now().normalize()
            days_diff = (today - last_date).days
            
            if days_diff <= 1:
                st.success(f"✅ 최신 데이터\n마지막: {last_date.strftime('%Y-%m-%d')}")
            elif days_diff <= 3:
                st.warning(f"⚠️ {days_diff}일 전 데이터\n(주말/휴장일 가능)")
            else:
                st.error(f"❌ {days_diff}일 전 데이터\n업데이트 필요!")
        
        with col2:
            missing = df['Close'].isna().sum()
            if missing == 0:
                st.success(f"✅ 결측치 없음")
            else:
                st.error(f"❌ 결측치 {missing}개")
        
        with col3:
            total_rows = len(df)
            expected = 252 * 10  # 10년 약 2520 거래일
            if total_rows >= expected * 0.9:
                st.success(f"✅ 충분한 데이터\n{total_rows}일")
            else:
                st.warning(f"⚠️ 데이터 부족?\n{total_rows}일")
        
        st.divider()
        
        # 수동 새로고침 버튼
        st.subheader("🔄 데이터 새로고침")
        st.caption("캐시를 무시하고 yfinance에서 새 데이터를 받아옵니다")
        
        if st.button("🔄 지금 새로고침", type="primary"):
            # 캐시 삭제
            cache_path = cache_dir / f"{ticker}.parquet"
            if cache_path.exists():
                cache_path.unlink()
            
            # 메타데이터에서 제거
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                if ticker in metadata:
                    del metadata[ticker]
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            # Streamlit 캐시도 클리어
            st.cache_data.clear()
            
            st.success("✅ 캐시 삭제 완료! 페이지를 새로고침하세요.")
            st.rerun()


if __name__ == "__main__":
    main()
