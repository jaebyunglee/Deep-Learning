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
      


# Autograph Warning 끄기 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.error) # Keras 코드 수행 시에 출력되는 Warning을 포함한 기타 메시지 제거
tf.autograph.set_verbosity(0) # autograph warning 제거
