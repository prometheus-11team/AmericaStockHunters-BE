from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

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

@app.post("/api/v1/start") # fastapi는 POST 방식일 때 자동으로 request body에서 데이터를 추출
def create_user(data: TradeRequest):
    """ 
    Create a new user with the provided data.
    This endpoint is used to start trading with the given parameters.
    """
    print("Received trading request:", data)
    # 실제 트레이딩 로직을 추가 필요
    
    return {"message": f"{data.name} 의 트레이딩이 시작됨!"}