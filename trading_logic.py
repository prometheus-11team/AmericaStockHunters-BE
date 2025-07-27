import pandas as pd
import numpy as np
from stable_baselines3 import TD3
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.agents.stablebaselines3.models import DRLAgent
from fastapi import HTTPException
import logging

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


def trading_pipeline(name, model_save_path, data_path, initial_capital, start_date, end_date):
    try:
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
        df = filter_data_by_date(df, start_date, end_date)
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

        trading_result = {
            "name": name,
            "initial_capital": initial_capital,
            "final_asset": final_asset,
            "profit": final_asset - initial_capital,
            "profit_rate": round((final_asset / initial_capital - 1) * 100, 2),
            "sharpe_ratio": sharpe_ratio,
            "account_values": df_account_value.to_dict(orient='records'),  # 포트폴리오 가치 시계열
            "trades": df_actions.to_dict(orient='records')                # 개별 매매 기록
        }
        logger.info("Trading completed successfully!")
        logger.info(f"Trading result: {trading_result}")

        return trading_result
        
    except Exception as e:
        logger.error(f"Error in trading_pipeline: {str(e)}", exc_info=True)
        raise e
