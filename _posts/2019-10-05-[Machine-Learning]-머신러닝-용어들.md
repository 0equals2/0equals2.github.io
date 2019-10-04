---
layout: post
title: "[Machine Learning] 머신러닝 용어들"
excerpt: "machine learning termimogy "
categories: ds
tags: ml
---





## 머신러닝 용어

Input Data 기본적으로 숫자. 이미지, 동영상 등이 주어져도 이것을 어떻게 숫자로 바꿀지를 고민하게 된다.

Output/Class/Label 결과값을 one-hot coding으로 해당 class에 맞게 출력

Training / Learning 여러 최적화 methods를 사용해서 weight를 찾아가는 과정

Dataset 기본적으로 인풋, 아웃풋으로 구성됨. 

​	- training data 학습에 사용하는 데이터. 

​	- validation 하이퍼 파라미터를 잡을 때 사용하는 데이터.

​	- test data 테스트 데이터는 모델 학습에 절대 관여하지 않는다.

Neuron 뉴런

Basic single layer network = Fully connet layer = Dense layer : non linear mapping을 찾는 도구

Neural Network 여러 개의 layer가 있는 것 (activation fuction이 없으면 hidden layer 가 100개든 1000개든 같다.)

Activation Function : non linearity를 주기 위한 것.

sigmoid 0에서 1로 떨어뜨려야 할 때

thanh -1 에서 1로

relu 일반적인 분류에서 좋은 성능을 보임

softplus 회귀에서 유용

one Epoch 전체 데이터를 한 번 사용했을 때까지의 단위

Batch size  전체 데이터를 사용할 수 없을 때, 한 번 계산할 때의 사이즈

one iteration 전체 데이터를 배치 사이즈로 나눈 것 (몇 번 업데이트를 했는지)

​		55000개의 트레이닝 데이터가 있을 때, 배치 사이즈가 1000일 경우, 55번 이터레이션으로 한 에포크가 된다.

cost function 줄이고 싶은 어떤 것. 목표는 같지만 cost function은 다를 수 있다. (무엇보다 미분이 가능해야 한다. 미분이 안되면 강화학습 알고리즘을 활용하면 해소할 수 있다.)

MNIST 계속 보게 될 데이터셋 (mixed national Institute of Standards and Technology)