// 시그모이드 활성화 함수
function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

// 로그 손실(Cross-Entropy) 함수의 경사(Gradient) 계산
// 입력값(x), 실제값(yTrue), 현재 가중치(weight)를 받아 기울기를 반환
function calculateLogGradient(x, yTrue, weight) {
    const z = x * weight;
    const yPred = sigmoid(z);
    
    // 로그 손실 함수의 미분 결과 (예측 오차 * 입력값)
    const gradient = (yPred - yTrue) * x;
    
    return {
        prediction: yPred,
        gradient: gradient
    };
}

// 예시 실행
const xInput = 2.0;
const actualValue = 1.0;
const currentWeight = 0.5;

const result = calculateLogGradient(xInput, actualValue, currentWeight);
console.log("예측값:", result.prediction);
console.log("계산된 경사(기울기):", result.gradient);
