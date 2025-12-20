"""데이터 수집 모듈"""

import yfinance as yf
import pandas as pd
from typing import Optional, List, Dict
from datetime import datetime


class DataFetcher:
    """주식 데이터 수집 클래스"""
    
    def __init__(self, tickers: List[str]):
        """
        Args:
            tickers: 종목 티커 리스트 (예: ["QQQ", "SPY"])
        """
        self.tickers = tickers
    
    def fetch(self, period: str = "10y") -> Dict[str, pd.DataFrame]:
        """
        여러 종목 데이터 수집
        
        Args:
            period: 데이터 기간 (예: "10y", "5y", "1y")
        
        Returns:
            {ticker: DataFrame} 딕셔너리
        """
        data = {}
        
        for ticker in self.tickers:
            df = self.fetch_single(ticker, period)
            if df is not None and not df.empty:
                data[ticker] = df
        
        return data
    
    def fetch_single(self, ticker: str, period: str = "10y") -> Optional[pd.DataFrame]:
        """
        단일 종목 데이터 수집
        
        Args:
            ticker: 종목 티커
            period: 데이터 기간
        
        Returns:
            OHLCV 데이터프레임 또는 None
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            
            if df.empty:
                print(f"⚠️  {ticker}: 데이터 없음")
                return None
            
            # 컬럼 정리 (필요한 것만)
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            available_cols = [col for col in required_cols if col in df.columns]
            df = df[available_cols].copy()
            
            # 인덱스를 timezone-naive datetime으로 변환
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = 'Date'
            
            return df
            
        except Exception as e:
            print(f"❌ {ticker} 데이터 수집 오류: {e}")
            return None
    
    def fetch_incremental(self, ticker: str, last_date: datetime) -> Optional[pd.DataFrame]:
        """
        증분 데이터 수집 (마지막 날짜 이후만)
        
        Args:
            ticker: 종목 티커
            last_date: 기존 데이터의 마지막 날짜
        
        Returns:
            새로운 데이터프레임 또는 None
        """
        try:
            stock = yf.Ticker(ticker)
            
            # 마지막 날짜 다음날부터 오늘까지
            start_date = last_date + pd.Timedelta(days=1)
            end_date = datetime.now()
            
            if start_date >= end_date:
                return None  # 새 데이터 없음
            
            df = stock.history(start=start_date.strftime('%Y-%m-%d'),
                              end=end_date.strftime('%Y-%m-%d'))
            
            if df.empty:
                return None
            
            # 컬럼 정리
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            available_cols = [col for col in required_cols if col in df.columns]
            df = df[available_cols].copy()
            
            # timezone 처리
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = 'Date'
            
            return df
            
        except Exception as e:
            print(f"❌ {ticker} 증분 데이터 수집 오류: {e}")
            return None

