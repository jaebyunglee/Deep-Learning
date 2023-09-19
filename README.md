# GRU

TimeDistributed
- 2번째 dim을 time으로 보고 레이어 적용
- ex) (배치,time,width,length,channel)에 time별로 conv2d 적용

TimeDistributed Dense(1)
- (배치,time,feature) 2번째 dim을 time으로 보고 각 time에 Dense 적용
- ex) input shape : (배치,time,feature) => (배치,time,1)

Dense
- 마지막 dim에 Dense(1) 레이어 적용
- ex) 
- input shape : (배치,128) => (배치,1)
- input shape : (배치,128,128) => (배치,128,1)
- input shape : (배치,time,feature) => (배치,time,1)
      


# Autograph Warning 끄기 (둘다 적용해야 꺼짐)
- tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # Keras 코드 수행 시에 출력되는 Warning을 포함한 기타 메시지 제거
- tf.autograph.set_verbosity(0) # autograph warning 제거

# Model 계산 과정에서 print 해보기
compile에 run_eagerly=True 추가 !!
custom loss나 custom metric 만들 때 계산 과정을 볼 수 있음

### 1) example code
def custom loss(y_true, y_pred):
    print(y_true)
    print(y_pred)
    score = .......
    return score



