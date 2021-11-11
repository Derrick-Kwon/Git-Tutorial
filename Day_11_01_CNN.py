#CNN 기본 basic
#정답은 원-핫 벡터를 이용한다
# 모든 절차 ABCD 중 A는 B를, B는C를 C는 D를, ABC는 Dense를 위해서 존재한다(앞쪽에 있는것은 뒤쪽에 전달할 featrue을 뽑는다)
# cnn 에서 ABC 층은 Dense를 위한 양질의 feature을 생성한다!
# CNN은 사진데이터에서 feature을 뽑는다. - feature extraction
# subsampling layers /a fully connected layer 을 교차해서 만든다
# 필터와 커널은 같은 말이다!
# 필터는 행렬곱! 그리고 일치하는 부분이 많으면 그부분 만 값이 커진다. 이거로 유사성을 알수있다.
# 데이터를 많이 넣을수록 양질의 feature를 뽑을 수 있다.


#convolution (곱셈) filter을 슬라이딩시켜가면서 곱한다

#필터연산 1번은 75 = 5x5(filter의크기)x3(rgb) 를 곱해주는걸 1번으로 친다.

#스트라이드 슬라이딩을 1번씩이 아니라 2~3번씩 가는거다.

#필터를 여기서 weight라고 부른다.

#학습은 weight를 업데이트 하는것이다 따라서 weight은 바뀌어야하고, filter도 계속해서 바뀌어야 한다.

#6filter을 뽑아내는것 = 6개층의 activation map/ image 또한 행렬곱해야하므로 depth가 6이어야 한다. 이 각각의 층의
#depth를 channel이라고 한다!!!!!! (중요***)

#496: convolution으로 연산을 하면 자연스럽게 테두리는 1번씩 연산되고 가운데 있는것은 가장 많이 연산에
#참가하게 된다! 따라서 중요도를 자연스럽게 나뉘게 된다.
#패딩을 씌우게 되면 중요도를 부여하면서 크기가 줄어들지 않는 이중적인 효과가 있다!
#499: sub-sampling convolution연산이 패딩을 사용하면 크기는 줄어들지 않는 대신 두꺼워진다. # maxpooling : 가장 큰 숫자를 뽑는다. /평균 pooling 평균값을 얻어낸다.

#502: maxpooling 은 곱셈을 하지 않고, 그냥 특정 레이어에서 값을 뽑아내기만 하므로 layer라고 하지는 않는다
#maxpooling 을 쓰는 이유? 두드러지는 특징을 부각하기 위해서이다.
#pooling 을 지나갈때마다 1/2로 크기가 줄어든다. 일반적인 모델은 pooling 을 5번 지나간다.


