from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# 전역 가중치 상태
model_state = {
    "weight": 0.5,
    "learning_rate": 0.1
}

class GradientData(BaseModel):
    gradient: float

@app.post("/update")
def update_weights(data: GradientData):
    # 자바스크립트가 보낸 경사값(gradient)을 이용해 가중치 업데이트
    global model_state
    model_state["weight"] -= model_state["learning_rate"] * data.gradient
    
    return {
        "status": "success",
        "new_weight": model_state["weight"]
    }

@app.get("/weight")
def get_weight():
    return model_state
