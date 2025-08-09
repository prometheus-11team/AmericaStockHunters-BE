import pandas as pd
import numpy as np
from stable_baselines3 import TD3

import sys
sys.path.append('/Users/jisu/Desktop/dev/prometheus/Stock/prometheus-11team/FinRL-Library') # main 브랜치 (250802)
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.agents.stablebaselines3.models import DRLAgent

from fastapi import HTTPException
import logging
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def load_model(model_save_path):
    # TD3 모델을 직접 로드
    model = TD3.load(model_save_path)
    print("model loaded successfully")
    return model


def prepare_env(df, initial_capital):
    USED_TICS = ['AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT', 'NVDA', 'TSLA']
    print(USED_TICS)

    # BEFOR_EXPENDED_INDICATORS = [
    #     'high', 'low', 'open', 'volume', 'SMA_50', 'SMA_200', 'RSI_14', 'ROC_10', 'MACD', 'MACD_Signal',
    #     'Federal Funds Rate', '10Y Treasury Yield', 'CPI', 'Core CPI', 'PCE Price Index',
    #     'Retail Sales', 'Unemployment Rate', 'Non-Farm Payrolls', 'M2 Money Stock'
    # ]

    ALL_INDICATORS = [ # 기술적 지표 + macro + 재무정보 + 감성분석 + Transformer
        'SMA_50', 'SMA_200', 'RSI_14', 'ROC_10', 'MACD', 'MACD_Signal', 'Federal Funds Rate', 
        '10Y Treasury Yield', 'CPI', 'Core CPI', 'PCE Price Index', 'Retail Sales', 'Unemployment Rate', 
        'Non-Farm Payrolls', 'M2 Money Stock', 'transformer_prediction', 'transformer_confidence', 
        'transformer_signal', 'negative_google_news', 'negative_reddit', 'neutral_google_news', 
        'neutral_reddit', 'positive_google_news', 'positive_reddit']
    # 실제 존재하는 컬럼만 남김
    ALL_INDICATORS = [col for col in ALL_INDICATORS if col in df.columns]
    print(ALL_INDICATORS)
    
    df = df[df['tic'].isin(USED_TICS)]
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)
    df = df.ffill().bfill()
    df = df.dropna(subset=ALL_INDICATORS)
    df.index = df['date'].factorize()[0]
    stock_dim = len(USED_TICS)
    state_space = 1 + 2 * stock_dim + len(ALL_INDICATORS) * stock_dim
    logger.info(f"Stock dimension: {stock_dim}")
    logger.info(f"All indicators: {len(ALL_INDICATORS)} \n{ALL_INDICATORS}")
    logger.info(f"State space size: {state_space}")
    env_config = {
        "hmax"  : 800,  # simulation_from_saved_testset.py와 일치
        "initial_amount" : initial_capital,
        "buy_cost_pct" : [0.0005] * stock_dim,  # simulation_from_saved_testset.py와 일치
        "sell_cost_pct" : [0.001] * stock_dim,
        "state_space" : state_space,
        "stock_dim" : stock_dim,
        "tech_indicator_list" : ALL_INDICATORS,
        "action_space" : stock_dim,
        "reward_scaling" : 2e-1,  # simulation_from_saved_testset.py와 일치
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
    df_actions = df_actions.reset_index(drop=True) # 인덱스 리셋 추가
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
    df_actions = df_actions.reset_index(drop=True)  # 인덱스 리셋 추가
    """
    df_actions를 프론트엔드 TradingHistory 페이지에 필요한 거래내역 형식으로 변환
    """
    transactions = []
    USED_TICS = ['AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT', 'NVDA', 'TSLA']
    
    if df_actions.empty:
        logger.info("df_actions is empty, no transactions to format")
        return []
    
    logger.info(f"df_actions columns: {df_actions.columns.tolist()}")
    logger.info(f"df_actions shape: {df_actions.shape}")
    
    # 각 종목별 가격 데이터 준비
    price_data = {}
    try:
        for tic in USED_TICS:
            tic_data = df_data[df_data['tic'] == tic]
            if not tic_data.empty:
                price_data[tic] = tic_data
    except Exception as e:
        logger.error(f"Error preparing price data: {e}")
        return []
    
    # 각 종목별 평균 매입 단가 추적
    avg_purchase_prices = {tic: 0.0 for tic in USED_TICS}
    total_shares_held = {tic: 0 for tic in USED_TICS}
    total_cost_basis = {tic: 0.0 for tic in USED_TICS}
    
    transaction_id = 1
    
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
                
                # 해당 시점의 가격 정보 가져오기
                price = 100.0  # 기본값
                if tic in price_data:
                    tic_price_data = price_data[tic]
                    if idx < len(tic_price_data):
                        price = tic_price_data.iloc[idx]['close']
                    else:
                        # 인덱스가 범위를 벗어나면 마지막 가격 사용
                        price = tic_price_data.iloc[-1]['close']
                
                # 거래 타입 결정 (양수면 매수, 음수면 매도)
                trade_type = "Buy" if action_value > 0 else "Sell"
                
                # 거래 금액 계산
                trade_amount = quantity * price
                
                # 손익 계산 (매도인 경우)
                profit_loss = None
                if trade_type == "Sell":
                    # 평균 매입 단가가 있는 경우에만 손익 계산
                    if avg_purchase_prices[tic] > 0:
                        # 매도 시 손익 = (매도가격 - 평균매입단가) * 매도수량
                        profit_loss = (price - avg_purchase_prices[tic]) * quantity
                        profit_loss = round(profit_loss)
                        logger.info(f"Sell profit/loss for {tic}: (${price:.2f} - ${avg_purchase_prices[tic]:.2f}) * {quantity} = ${profit_loss:.2f}")
                    else:
                        profit_loss = 0.0
                        logger.warning(f"No average purchase price available for {tic}, setting profit/loss to 0")
                
                # 평균 매입 단가 업데이트 (매수인 경우)
                if trade_type == "Buy":
                    # 새로운 매수로 인한 평균 매입 단가 재계산
                    current_shares = total_shares_held[tic]
                    current_cost = total_cost_basis[tic]
                    
                    new_shares = quantity
                    new_cost = quantity * price
                    
                    total_shares_held[tic] = current_shares + new_shares
                    total_cost_basis[tic] = current_cost + new_cost
                    
                    # 평균 매입 단가 = 총 매입 비용 / 총 보유 주식 수
                    if total_shares_held[tic] > 0:
                        avg_purchase_prices[tic] = total_cost_basis[tic] / total_shares_held[tic]
                        logger.info(f"Updated avg purchase price for {tic}: ${avg_purchase_prices[tic]:.2f} (shares: {total_shares_held[tic]}, cost: ${total_cost_basis[tic]:.2f})")
                
                # 매도 시 보유 주식 수 감소
                elif trade_type == "Sell":
                    total_shares_held[tic] -= quantity
                    # 매도 후 보유 주식이 0이 되면 평균 매입 단가 초기화
                    if total_shares_held[tic] <= 0:
                        avg_purchase_prices[tic] = 0.0
                        total_cost_basis[tic] = 0.0
                        total_shares_held[tic] = 0
                        logger.info(f"Reset avg purchase price for {tic} after selling all shares")
                
                transaction = {
                    "id": int(transaction_id),
                    "datetime": f"{date_str} 10:00",
                    "symbol": tic,
                    "type": trade_type,
                    "quantity": int(quantity),
                    "price": float(round(price, 2)),
                    "amount": float(round(trade_amount, 2)),
                    "profitLoss": float(profit_loss) if profit_loss is not None else None,
                    "day": int(idx + 1)
                }
                
                transactions.append(transaction)
                logger.info(f"Transaction {transaction_id}: {trade_type} {quantity} shares of {tic} at ${price:.2f} = ${trade_amount:.2f}")
                transaction_id += 1
    
    logger.info(f"Total transactions formatted: {len(transactions)}")
    return transactions


def calculate_portfolio_status(df_actions, df_data, df_account_value, initial_capital):
    """
    투자 종료일 기준 보유중인 종목들의 포트폴리오 상태를 계산
        - 매수: 수량/원가 증가
        - 매도: 보유평단 * 매도수량 만큼 원가 차감 (MA 방식)
        - 날짜 기준으로 액션-가격 매칭
    """
    USED_TICS = ['AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT', 'NVDA', 'TSLA']
    portfolio_status = []

    # 0) 날짜 인덱스 준비 (액션에 날짜가 없으므로 account_value의 date를 사용)
    #    df_actions의 길이와 거래일 수가 불일치하면 뒤에서 안전하게 슬라이스 됩니다.
    trade_dates = None
    if 'date' in df_account_value.columns:
        trade_dates = pd.to_datetime(df_account_value['date'].values)
    elif 'date' in df_data.columns:
        # 종목/날짜 중복이 있으므로 unique
        trade_dates = pd.to_datetime(df_data['date'].unique())
        trade_dates.sort_values(inplace=True)

    # df_actions에 date 부여
    actions = df_actions.copy()
    if trade_dates is not None:
        actions = actions.iloc[:len(trade_dates)].copy()
        actions.insert(0, 'date', trade_dates[:len(actions)])
        actions.set_index('date', inplace=True)
    # (그래도 날짜가 없다면 RangeIndex로 동작하지만, 아래 join에서 의미가 줄어듭니다)

    # df_data: 종가 시계열 pivot (행: date, 열: 종목)
    prices = (
        df_data.loc[:, ['date', 'tic', 'close']]
               .assign(date=lambda x: pd.to_datetime(x['date']))
               .pivot(index='date', columns='tic', values='close')
               .sort_index()
    )

    # 액션과 가격을 날짜 기준으로 정렬/정합
    # (액션이 더 짧거나 길면 공통 구간만 사용)
    if not actions.empty:
        acts = actions.join(prices, how='left', rsuffix='_price')
    else:
        acts = prices.copy()
        for t in USED_TICS:
            if t not in acts.columns:
                acts[t] = 0  # 액션이 없으면 0

    # 결측 가격은 이전가로 보간(휴장일/결측 방지)
    acts = acts.sort_index()
    acts.loc[:, prices.columns] = acts.loc[:, prices.columns].ffill()

    # 종목별 보유/원가 누적
    final_total_assets = initial_capital
    if not df_account_value.empty:
        if 'account_value' in df_account_value.columns:
            final_total_assets = float(df_account_value.iloc[-1]['account_value'])
        elif 'total_assets' in df_account_value.columns:
            final_total_assets = float(df_account_value.iloc[-1]['total_assets'])

    for symbol in USED_TICS:
        if symbol not in acts.columns:  # 액션 없으면 스킵
            continue

        # 액션 수량과 가격 컬럼 준비
        qty_series = acts[symbol].fillna(0)
        # 가격은 prices에 동일 심볼이 있어야 합니다.
        if symbol in prices.columns:
            price_series = acts[symbol if symbol in prices.columns else f'{symbol}_price']
            # 위 줄은 방어적이지만, 일반적으로 symbol가 prices.columns에 존재
        else:
            # 해당 종목 데이터 없으면 스킵
            continue

        total_shares = 0
        total_cost = 0.0

        for qty, px in zip(qty_series.values, price_series.values):
            if np.isnan(qty) or np.isnan(px) or qty == 0:
                continue

            if qty > 0:
                # --- 매수 ---
                buy_qty = int(qty)
                # 수수료/슬리피지 반영 예:
                # eff_buy_px = px * (1 + buy_fee_pct)
                eff_buy_px = px
                total_shares += buy_qty
                total_cost   += buy_qty * eff_buy_px

            else:
                # --- 매도 ---
                sell_qty = min(int(-qty), total_shares)  # 과매도 방지
                if sell_qty > 0 and total_shares > 0:
                    avg_px = total_cost / total_shares
                    # 원가에서 '보유평단 * 매도수량'만큼 차감
                    total_shares -= sell_qty
                    total_cost   -= avg_px * sell_qty
                    # 현금 유입/수수료는 원한다면 별도로 추적:
                    # cash += sell_qty * px * (1 - sell_fee_pct)

        # 최종 보유가 있으면 상태 산출
        if total_shares > 0:
            current_price = prices[symbol].iloc[-1]
            avg_price = total_cost / total_shares
            total_value = total_shares * current_price
            profit_rate = ((current_price / avg_price) - 1.0) * 100.0 if avg_price > 0 else 0.0
            share_percentage = (total_value / final_total_assets * 100.0) if final_total_assets > 0 else 0.0

            portfolio_status.append({
                "symbol": symbol,
                "avg": round(float(avg_price), 2),
                "now": round(float(current_price), 2),
                "profit_rate": round(float(profit_rate), 2),
                "total": round(float(total_value), 2),
                "share": round(float(share_percentage), 2),
                "qty": int(total_shares),
                "profit": round(float((current_price - avg_price) * total_shares), 2),
            })

    return portfolio_status


def trading_pipeline(name, model_save_path, df, initial_capital, start_date, end_date):
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
        
        logger.info(f"initial_capital: {initial_capital}")
        logger.info(f"start_date: {start_date}")
        logger.info(f"end_date: {end_date}")
        logger.info(f"Starting trading pipeline with parameters:")
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
        logger.info("Data preprocessing completed.")

        # 환경을 먼저 생성한 후 모델 로드
        logger.info(f"model_save_path: {model_save_path}")
        logger.info("Preparing environment...")
        env = prepare_env(df, initial_capital)
        logger.info("Environment prepared successfully.")
        
        logger.info("Loading model...")
        model = load_model(model_save_path)
        logger.info("Model loaded successfully.")
        
        logger.info("Starting trading simulation...")
        df_account_value, df_actions = simulate_trading(model, env)
        logger.info("Trading simulation completed.")
        
        # df_actions 구조 로깅
        logger.info(f"df_actions shape: {df_actions.shape}")
        logger.info(f"df_actions columns: {df_actions.columns.tolist()}")
        if not df_actions.empty:
            logger.info(f"df_actions head: {df_actions.head().to_dict()}")
            logger.info(f"df_actions sum: {df_actions.sum().to_dict()}")

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
            logger.info("No transactions available")
            transactions = []

        # 포트폴리오 상태 계산
        logger.info("Calculating portfolio status...")
        try:
            portfolio_status = calculate_portfolio_status(df_actions, df, df_account_value, initial_capital)
            logger.info(f"Calculated portfolio status for {len(portfolio_status)} stocks")
        except Exception as e:
            logger.error(f"Error calculating portfolio status: {e}")
            portfolio_status = []

        trading_result = {
            "name": name,
            "initial_capital": initial_capital,
            "final_asset": final_asset,
            "profit": final_asset - initial_capital,
            "profit_rate": round(((final_asset - initial_capital) / initial_capital) * 100, 2),
            "sharpe_ratio": sharpe_ratio,
            "account_values": df_account_value.to_dict(orient='records'),  # 포트폴리오 가치 시계열
            "transactions": transactions,                                    # 거래내역
            "portfolio_status": portfolio_status                             # 포트폴리오 상태
        }
        logger.info("Trading completed successfully!")
        logger.info(f"Trading result: {trading_result}")

        if not transactions:
            logger.info("No transactions occurred during the simulation.")
        if not portfolio_status:
            logger.info("No stocks held at the end of the simulation.")

        return trading_result
        
    except Exception as e:
        logger.error(f"Error in trading_pipeline: {str(e)}", exc_info=True)
        raise e
