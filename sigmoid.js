// 시그모이드 함수
function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

// 로그 손실 기반 경사 계산 후 파이썬 서버로 전송
async function stepAndSync(x, yTrue, currentWeight) {
    const z = x * currentWeight;
    const yPred = sigmoid(z);
    
    // 로그 손실 함수의 경사 계산
    const gradient = (yPred - yTrue) * x;
    
    console.log(`[JS] 계산된 예측값: ${yPred}, 경사: ${gradient}`);

    // 파이썬 서버(trainer.py)로 경사 값 전송
    try {
        const response = await fetch('http://localhost:8000/update', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ gradient: gradient })
        });
        
        const result = await response.json();
        console.log(`[Python 서버 응답] 갱신된 가중치: ${result.new_weight}`);
    } catch (error) {
        console.error("서버 통신 실패:", error);
    }
}

// 테스트 실행 (x=2.0, 실제값=1.0, 현재 가중치=0.5)
stepAndSync(2.0, 1.0, 0.5);
