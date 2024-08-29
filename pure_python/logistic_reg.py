import numpy as np

# 신인 농구 선수의 NBA 드래프트 결과 데이터
# 각 행별로 평균 득점, 리바운드, 어시스트, 드래프트 여부(성공 1, 실패 0 -> 정답 레이블)
X = np.array([[12, 3, 6, 0],
              [13, 4, 4, 1],
              [13, 4, 6, 0],
              [12, 9, 9, 1],
              [14, 4, 5, 1],
              [14, 4, 4, 0],
              [17, 2, 2, 0],
              [17, 6, 5, 1],
              [21, 5, 7, 1],
              [21, 9, 3, 0],
              [24, 11, 11, 1],
              [24, 4, 5, 0]])

W = np.random.randn(3, 1)
B = np.random.randn()
# 일반적인 LR인 0.01을 사용할 경우 Loss 값이 진동하게 된다.
LEARNING_RATE = 0.001

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

# cross entropy error
def loss(Y, P):
    return np.sum(-Y*np.log(P) - (1-Y)*np.log(1-P))

def logistic_forward(X, Y, W, B):
    Z = np.dot(X, W) + B
    A = sigmoid(Z)
    L = loss(Y, A)
    return Z, A, L

def loss_gradient(X, Y, W, B):
    Z, A, L = logistic_forward(X, Y, W, B)
    # print('Y shape: ', Y.shape)
    # print('A shape: ', A.shape)
    # print('L: ', L)
    # print('='*50)

    # partial L / partial A(y hat)
    dL_dA = -(Y / A) + ((1-Y) / (1-A))
    # partial A / partial Z
    dA_dZ = A - A**2
    dZ_dW = X.T
    #dZ_dB = 1

    # Chain rule로 구한 gradient
    dL_dW = np.dot(dZ_dW, dL_dA * dA_dZ)
    dL_dB = np.sum(dL_dA*dA_dZ, axis=0)

    # print('dL_dA shape: ', dL_dA.shape) # (12 ,)
    # print('dZ_dW shape: ', dZ_dW.shape) # (3, 12)
    # print('dL_dW shape: ', dL_dW.shape) # (3, )
    # print('dA_dZ value: ', dA_dZ)
    # print('dL_dB value: ', dL_dB)
    # print('=' * 50)
    return dL_dW, dL_dB, L

if __name__ == '__main__':
    input = X[:, 0:3]
    target = np.array(X[:, 3:]) # (12, 1) matrix를 얻는다. 여기서 뭔가 잘못해서 (12, ) vector를 얻게 되면 결과가 이상해진다..
    # print('input shape: ', input.shape)
    # print('target shape: ', target.shape)
    print('Before W: ', W)
    print('Before B: ', B)
    print('=' * 50)

    # 500회 train
    for i in range (500):
        # gradient descent
        dL_dW, dL_dB, L = loss_gradient(input, target, W, B)
        W = W - (LEARNING_RATE * dL_dW)
        B = B - (LEARNING_RATE * dL_dB)
        print('train', i+1, ', loss: ', L)

    print('After W: ', W)
    print('After B: ', B)

'''
1. (3, ) * (1, 3) 을 하면 (1, 3) matrix를 얻는데,
    (3, ) * (3, 1)을 하면 (3, 3) matrix를 얻는 이유
    -> element-wise 연산을 하는 과정에서 broadcasting이 적용됨
    
2. LR 값을 0.01(가장 일반적인 값)로 할 경우 Loss값이 진동하게 되는데 그 이유가 무엇일까?
-> guess 1. 데이터 수가 너무 적어서?
-> guess 2. 첫 W, b 값을 randn으로 정해서? 
'''