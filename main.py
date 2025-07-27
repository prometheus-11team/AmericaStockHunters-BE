from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from trading_logic import trading_pipeline
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) # (reload=True) WARNING:  You must pass the application as an import string to enable 'reload' or 'workers'.
