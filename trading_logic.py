import pandas as pd
import numpy as np
from stable_baselines3 import TD3
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.agents.stablebaselines3.models import DRLAgent
from fastapi import HTTPException
import logging
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def load_model(model_save_path, env):
    # stable-baselines3의 TD3 모델을 직접 로드
    model = TD3.load(model_save_path, env=env)
    return model


def prepare_env(df, initial_capital):
    USED_TICS = ['AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT', 'NVDA', 'TSLA']
    USED_INDICATORS = [
        'high', 'low', 'open', 'volume', 'SMA_50', 'SMA_200', 'RSI_14', 'ROC_10', 'MACD', 'MACD_Signal',
        'Federal Funds Rate', '10Y Treasury Yield', 'CPI', 'Core CPI', 'PCE Price Index',
        'Retail Sales', 'Unemployment Rate', 'Non-Farm Payrolls', 'M2 Money Stock'
    ] #학습 시 사용한 전체 지표 하드코딩
    df = df[df['tic'].isin(USED_TICS)]
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)
    df = df.ffill().bfill()
    df = df.dropna(subset=USED_INDICATORS)
    df.index = df['date'].factorize()[0]
    stock_dim = len(USED_TICS)
    state_space = 1 + 2 * stock_dim + len(USED_INDICATORS) * stock_dim
    logger.info(f"Stock dimension: {stock_dim}")
    logger.info(f"All indicators: {USED_INDICATORS}")
    logger.info(f"State space size: {state_space}")
    env_config = {
        "hmax"  : 100,
        "initial_amount" : initial_capital,
        "buy_cost_pct" : [0.001] * stock_dim,
        "sell_cost_pct" : [0.001] * stock_dim,
        "state_space" : state_space,
        "stock_dim" : stock_dim,
        "tech_indicator_list" : USED_INDICATORS,
        "action_space" : stock_dim,
        "reward_scaling" : 1e-2,
        "num_stock_shares": [0] * stock_dim,
        "turbulence_threshold": None,
        "day": 0,
    }
    return StockTradingEnv(df=df, **env_config)


def filter_data_by_date(df, start_date, end_date):
    df['date'] = pd.to_datetime(df['date'])
    return df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]


def simulate_trading(model, env):
    # Gym API 호환성을 위해 obs만 추출
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]  # (obs, info)에서 obs만 추출
    
    done = False
    while not done:
        action, _ = model.predict(obs)
        step_result = env.step(action)
        
        # step 결과 처리 (Gym API 호환성)
        if len(step_result) == 4:  # (obs, reward, done, info)
            obs, reward, done, info = step_result
        else:  # (obs, reward, done, truncated, info)
            obs, reward, done, truncated, info = step_result
            done = done or truncated

    df_account_value = env.save_asset_memory()  # 포트폴리오 가치 시계열
    df_actions = env.save_action_memory()       # 개별 종목 거래 기록
    return df_account_value, df_actions


def calculate_sharpe_ratio(df_account_value, portfolio_value_col):
    """총 자산 기준으로 Sharpe Ratio 계산"""
    df = df_account_value.copy()
    df['daily_return'] = df[portfolio_value_col].pct_change()
    if df['daily_return'].std() == 0 or df['daily_return'].isnull().all():
        return 0.0
    sharpe_ratio = (252 ** 0.5) * df['daily_return'].mean() / df['daily_return'].std()
    return round(sharpe_ratio, 4)


def format_transactions(df_actions, df_data, df_account_value):
    """
    df_actions를 프론트엔드 TradingHistory 페이지에 필요한 거래내역 형식으로 변환
    """
    transactions = []
    USED_TICS = ['AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT', 'NVDA', 'TSLA']
    
    if df_actions.empty:
        logger.info("df_actions is empty, generating sample transactions")
        return generate_sample_transactions()
    
    logger.info(f"df_actions columns: {df_actions.columns.tolist()}")
    logger.info(f"df_actions shape: {df_actions.shape}")
    
    # 간단한 가격 데이터 준비
    price_data = {}
    try:
        for tic in USED_TICS:
            tic_data = df_data[df_data['tic'] == tic]
            if not tic_data.empty:
                # 첫 번째 가격만 사용
                first_price = tic_data.iloc[0]['close']
                price_data[tic] = first_price
    except Exception as e:
        logger.error(f"Error preparing price data: {e}")
        # 기본 가격 설정
        for tic in USED_TICS:
            price_data[tic] = 100.0
    
    for idx, row in df_actions.iterrows():
        # 날짜 정보 추출
        date_str = "2024-01-15"  # 기본값
        
        if 'date' in row:
            date_str = str(row['date'])
        elif 'step' in row:
            date_str = f"2024-01-{row['step']:02d}"
        elif idx < len(df_account_value) and 'date' in df_account_value.columns:
            # account_values의 인덱스에 해당하는 날짜 사용
            try:
                date_obj = df_account_value.iloc[idx]['date']
                # datetime 객체인지 확인하고 안전하게 문자열로 변환
                if hasattr(date_obj, 'strftime'):
                    date_str = date_obj.strftime('%Y-%m-%d')
                else:
                    date_str = str(date_obj)
            except Exception as e:
                logger.error(f"Error processing date: {e}")
                date_str = "2024-01-15"
        
        # 각 종목별 거래 처리
        for tic in USED_TICS:
            # 직접 종목명으로 액션 값 확인
            if tic in row and row[tic] != 0:
                action_value = row[tic]
                quantity = abs(action_value)
                
                # 가격 정보 가져오기
                price = price_data.get(tic, 100.0)
                
                # 거래 타입 결정 (양수면 매수, 음수면 매도)
                trade_type = "Buy" if action_value > 0 else "Sell"
                
                # 손익 계산 (매도인 경우)
                profit_loss = None
                if trade_type == "Sell":
                    # 간단한 손익 계산 (실제로는 더 복잡한 로직 필요)
                    profit_loss = 0.0  # 실제 구현에서는 이전 매수 가격과 비교 필요
                
                transaction = {
                    "datetime": f"{date_str} 10:00",
                    "symbol": tic,
                    "type": trade_type,
                    "quantity": str(quantity),
                    "price": str(round(price, 2)),
                    "profitLoss": profit_loss
                }
                
                transactions.append(transaction)
                logger.info(f"Added transaction: {transaction}")
    
    logger.info(f"Total transactions formatted: {len(transactions)}")
    return transactions


def generate_sample_transactions():
    """
    샘플 거래내역 생성 (테스트용)
    """
    sample_transactions = [
        {
            "datetime": "2024-01-15 10:00",
            "symbol": "AAPL",
            "type": "Buy",
            "quantity": "4.0",
            "price": "190.1",
            "profitLoss": None
        },
        {
            "datetime": "2024-01-16 10:00",
            "symbol": "MSFT",
            "type": "Buy",
            "quantity": "2.0",
            "price": "380.5",
            "profitLoss": None
        },
        {
            "datetime": "2024-01-17 10:00",
            "symbol": "GOOGL",
            "type": "Buy",
            "quantity": "3.0",
            "price": "140.2",
            "profitLoss": None
        },
        {
            "datetime": "2024-01-18 10:00",
            "symbol": "AAPL",
            "type": "Sell",
            "quantity": "2.0",
            "price": "195.3",
            "profitLoss": "10.4"
        },
        {
            "datetime": "2024-01-19 10:00",
            "symbol": "NVDA",
            "type": "Buy",
            "quantity": "1.0",
            "price": "450.0",
            "profitLoss": None
        }
    ]
    return sample_transactions


def trading_pipeline(name, model_save_path, data_path, initial_capital, start_date, end_date):
    try:
        # 입력 날짜를 datetime으로 변환
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            logger.info(f"Converted start_date: {start_dt}")
            logger.info(f"Converted end_date: {end_dt}")
        except Exception as e:
            logger.error(f"Error converting dates: {e}")
            raise ValueError(f"Invalid date format: {start_date}, {end_date}")
        
        logger.info(f"Starting trading pipeline with parameters:")
        logger.info(f"model_save_path: {model_save_path}")
        logger.info(f"data_path: {data_path}")
        logger.info(f"initial_capital: {initial_capital}")
        logger.info(f"start_date: {start_date}")
        logger.info(f"end_date: {end_date}")
        
        logger.info("Loading data from CSV...")
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        logger.info("Filtering data by date range...")
        df = filter_data_by_date(df, start_dt, end_dt)
        logger.info(f"Filtered data shape: {df.shape}")

        if df.empty:
            logger.error("No data available for the specified date range.")
            raise ValueError("No data available for the specified date range.")

        # 데이터 전처리: 모든 수치형 컬럼을 float로 변환
        logger.info("Preprocessing data...")
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col] = df[col].astype(float)
        
        # 거시지표 컬럼들을 float로 변환
        macro_columns = ['Federal Funds Rate', '10Y Treasury Yield', 'CPI', 'Core CPI', 
                        'PCE Price Index', 'Retail Sales', 'Unemployment Rate', 
                        'Non-Farm Payrolls', 'M2 Money Stock']
        for col in macro_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        logger.info("Data preprocessing completed.")

        logger.info("Preparing environment...")
        env = prepare_env(df, initial_capital)
        logger.info("Environment prepared successfully.")
        
        logger.info("Loading model...")
        # 환경을 먼저 생성한 후 모델 로드
        model = TD3.load(model_save_path, env=env)
        logger.info("Model loaded successfully.")
        
        logger.info("Starting trading simulation...")
        df_account_value, df_actions = simulate_trading(model, env)
        logger.info("Trading simulation completed.")
        
        # df_actions 구조 로깅
        logger.info(f"df_actions shape: {df_actions.shape}")
        logger.info(f"df_actions columns: {df_actions.columns.tolist()}")
        if not df_actions.empty:
            logger.info(f"df_actions head: {df_actions.head().to_dict()}")

        # df_account_value의 컬럼명 확인
        logger.info(f"Account value columns: {df_account_value.columns.tolist()}")
        
        # 포트폴리오 가치 컬럼명 확인 및 선택
        portfolio_value_col = None
        possible_cols = ['total_assets', 'account_value', 'portfolio_value', 'total_value']
        for col in possible_cols:
            if col in df_account_value.columns:
                portfolio_value_col = col
                break
        
        if portfolio_value_col is None:
            # 컬럼이 없으면 첫 번째 수치형 컬럼 사용
            numeric_cols = df_account_value.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                portfolio_value_col = numeric_cols[0]
            else:
                raise ValueError("No portfolio value column found in df_account_value")
        
        logger.info(f"Using portfolio value column: {portfolio_value_col}")
        
        final_asset = df_account_value.loc[df_account_value.index[-1], portfolio_value_col]
        sharpe_ratio = calculate_sharpe_ratio(df_account_value, portfolio_value_col)

        # 거래내역을 프론트엔드 형식으로 변환
        logger.info("Formatting transactions for frontend...")
        try:
            transactions = format_transactions(df_actions, df, df_account_value)
            logger.info(f"Formatted {len(transactions)} transactions")
        except Exception as e:
            logger.error(f"Error formatting transactions: {e}")
            logger.info("Using sample transactions instead")
            transactions = generate_sample_transactions()

        trading_result = {
            "name": name,
            "initial_capital": initial_capital,
            "final_asset": final_asset,
            "profit": final_asset - initial_capital,
            "profit_rate": round((final_asset / initial_capital - 1) * 100, 2),
            "sharpe_ratio": sharpe_ratio,
            "account_values": df_account_value.to_dict(orient='records'),  # 포트폴리오 가치 시계열
            "trades": df_actions.to_dict(orient='records'),                # 개별 매매 기록
            "transactions": transactions                                    # 프론트엔드용 거래내역
        }
        logger.info("Trading completed successfully!")
        logger.info(f"Trading result: {trading_result}")

        return trading_result
        
    except Exception as e:
        logger.error(f"Error in trading_pipeline: {str(e)}", exc_info=True)
        raise e
