# GRU

TimeDistributed
- 2번째 dim을 time으로 보고 레이어 적용
- ex) (배치,time,width,length,channel)에 time별로 conv2d 적용

TimeDistributed Dense
- (배치,time,feature) 2번째 dim을 time으로 보고 각 time에 Dense 적용
- ex) input shape : (배치,time,feature) => (배치,time,1)

Dense
- 마지막 dim에 Dense 레이어 적용
- ex) input shape : (배치,128) => (배치,1)
      input shape : (배치,128,128) => (배치,128,1)
      input shape : (배치,time,feature) => (배치,time,feature)
      
