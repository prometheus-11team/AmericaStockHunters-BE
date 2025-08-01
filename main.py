from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi import Query
from fastapi.middleware.cors import CORSMiddleware
from trading_logic import trading_pipeline
from datetime import datetime, timedelta
import yfinance as yf
from typing import Optional
import uvicorn
import logging

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS middleware to allow requests from the frontend (개발 중엔 모든 origin 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 origin 허용 (개발 중, 배포 시에는 특정 origin으로 제한이 권장됨)
    allow_credentials=True, # 쿠키와 같은 인증 정보를 허용
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

class TradeRequest(BaseModel):
    name: str
    initialCapital: int
    startDate: str
    endDate: str

@app.get("/")
def read_root():
    return {"message": "Welcome to America Stock Hunters Backend!"}

@app.post("/api/v1/trades") # fastapi는 POST 방식일 때 자동으로 request body에서 데이터를 추출
def run_trading(req: TradeRequest):
    try: 
        logger.info(f"Received trading request: {req}")
        logger.info("Starting trading pipeline...")
        
        result = trading_pipeline(
            name=req.name,
            model_save_path = "models/td3_model_with_macro.zip",
            data_path="data/test.csv",
            initial_capital=req.initialCapital,
            start_date=req.startDate,
            end_date=req.endDate
        )
        
        logger.info("Trading pipeline completed successfully")
        logger.info(f"Result: {result}")
        
        return {
            "status": "success",
            "message": f"{req.name} 의 트레이딩이 완료됨",
            "result": result
        }
    except Exception as e:
        logger.error(f"Error in run_trading: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }
    
@app.get("/api/v1/trades")
def get_trading_results():
    # 전역상태로 저장한 트레이딩 결과를 기반으로 처리
    return {"message": "Trading results retrieved successfully!"}


@app.get("/api/v1/nasdaq")
def get_nasdaq(
    startDate: Optional[str] = Query(None, description="YYYY-MM-DD"),
    endDate: Optional[str] = Query(None, description="YYYY-MM-DD")
    # startDate: str = Query(..., description="YYYY-MM-DD"),
    # endDate:   str = Query(..., description="YYYY-MM-DD")
):
    if not startDate:
        raise HTTPException(status_code=400, detail="startDate is required and must not be empty.")
    if not endDate:
        raise HTTPException(status_code=400, detail="endDate is required and must not be empty.")
    
    try:
        # endDate를 포함하도록 하루 추가
        # 빈 문자열이 아니라는 것이 보장되므로 여기서 fromisoformat 호출
        ed = datetime.fromisoformat(endDate) + timedelta(days=1)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid endDate format. Use YYYY-MM-DD.")

    try:
        # startDate도 마찬가지로 유효성 검사
        datetime.fromisoformat(startDate) # 실제 사용은 yf.download에서 하므로 변수 저장 불필요
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid startDate format. Use YYYY-MM-DD.")
    

    # endDate를 포함하도록 하루 추가
    # ed = datetime.fromisoformat(endDate) + timedelta(days=1)
    df = yf.download(
        "^IXIC",
        start=startDate,
        end=ed.strftime("%Y-%m-%d"),
        interval="1d",
        progress=False,
        auto_adjust=False
    )
    # 빈 DataFrame이면 빈 배열로 응답
    if df.empty:
        return {"status":"success","result":{"nasdaq_values":[]}}

    base = df["Close"].iloc[0]
    nasdaq_values = [
        {
            "date": idx.strftime("%Y-%m-%dT00:00:00"),
            "value": round((price / base - 1) * 100, 4)
        }
        for idx, price in zip(df.index, df["Close"])
    ]
    return {"status":"success","result":{"nasdaq_values":nasdaq_values}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) # (reload=True) WARNING:  You must pass the application as an import string to enable 'reload' or 'workers'.