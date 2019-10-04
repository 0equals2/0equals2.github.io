---
layout: post
title: "[Machine Learning] introduction"
excerpt: "introduction to Machine Learning"
categories: ds
tags: ml
---



# 소개

머신러닝은 데이터에서 지식을 추출하는 작업으로, **통계학, 인공지능 그리고 컴퓨터 과학**의 합작이다. **예측 분석**이나 **통계적 머신러닝**으로도 불린다. 여러 상업적 어플리케이션 외에도 오늘날 데이터 기반 연구에 큰 영향을 끼쳐왔다. 별 탐구, 새로운 행성, 새로운 미립자의 발견, DNA 서열 분석 등 다양한 과학 분야에도 적용되었다. 머신러닝이 왜 유명해졌고, 어떤 문제를 해결할 수 있는지 알아보자.

## 왜?
초창기엔 지능형 어플리케이션의 시스템은 **하드 코딩**된 if와 else명령으로 데이터를 처리했다.  
이런 시스템의 단점:
* 결정에 필요한 로직이 한 분야 혹은 작업에 국한, 작업이 바뀌면 전체 시스템을 다시 개발해야 함
* 규칙을 설계하려면 분야 전문가들이 내리는 결정 방식을 잘 알아야 함  

요즘엔 스마트폰이 이미지로부터 얼굴을 찾아낼 수 있지만, 2001년 이전에는 얼굴인식은 풀 수 없는 문제였다. 얼굴인식은 규칙을 직접 만드는 데 실패한 대표적 예시이다. 사람이 얼굴을 인식하는 방식과 컴퓨터가 인식하는 방식이 다르기 때문에, 디지털 이미지에서 얼굴 구성 요소를 일련의 규칙으로 표현하는 것은이 근본적으로 불가능하다. 그러나 머신 러닝으로 아주 많은 얼굴이미지를 제공하면 얼굴을 특정하는 요소를 찾아낼 수 있다

### 머신러닝으로 풀 수 있는 문제
**지도학습** : 이미 알려진 사례를 바탕으로 일반화된 모델을 만들어 의사 결정 프로세스를 자동화  
사용자는 알고리즘에 입력과 기대되는 출력을 제공  
알고리즘은 주어진 입력에서 원하는 출력을 만드는 방법 모색  
입력 데이터로부터 기대한 출력이 나오도록 가르치기 때문에, 입력과 출력으로부터 학습하는 머신러닝 알고리즘을 **지도 학습(supervised)** 알고리즘이라 한다.  
입출력 데이터는 종종 수작업이 필요하지만, 지도학습 알고리즘은 분석하기에 좋고 성능을 측정하기도 쉽다.  

**지도학습의 예**  

1) **편지봉투 손으로 쓴 우편번호 판별**   
*입력* : 손글씨 스캔이미지  
기대하는 *출력* : 우편번호 숫자

2) **의료영상 이미지에 기반한 종양 판단**  
*입력* : 이미지  
*출력* : 종양이 양성인지 아닌지  
모델 구축을 위해 의료 영상 데이터베이스와 전문가의 의견이 필요  

3) **의심되는 신용카드 거래 감지**  
*입력* : 신용카드 거래 내역  
*출력* : 부정 거래 여부  

**비지도 학습** :알고리즘에 입력은 주어지지만 **출력은 제공되지 않는다.** 성공사례는 많지만 비지도 학습을 이해하거나 평가하기가 어렵다.

**비지도학습의 예**  

1) **블로그 글의 주제 구분**  
입력 : 블로그 글 (텍스트 데이터)   
사전에 어떤 주제인지, 몇 개의 주제가 있는지 알 수 없다.  

2) **취향이 비슷한 고객 그룹으로 묶기**  
쇼핑 사이트라면 부모, 독서광, 게이머 등의 그룹으로 묶을 수 있다. 어떤 그룹이 있는지 미리 알 수 없고 얼마나 많은 그룹이 있는지 알 수 없으므로 출력을 가지고 있지 않다.  

3) **비정상적인 웹사이트 접근 탐지**  
일상적이지 않은 접근 패턴을 찾으면 부정행위나 버그를 구별하는데 도움이 된다. 각각의 비정상 패턴은 서로 다를 수 있고 이미 가지고 있는 비정상 데이터도 없을 수 있다. 웹 트래픽만을 관찰 가능, 어떤 것이 정상인지 비정상인지 알지 못하므로 비지도 학습  

>! point : 지도 학습, 비지도 학습 모두 **데이터 준비가 중요** (컴퓨터가 **인식 가능**한)  
대부분 행과 열로 구성된 **테이블 형태**  (엑셀과 같이)  
행 : 판별해야 할 각각의 데이터 (개개의 이메일, 고객, 거래)    
열 : 데이터를 구성하는 각 속성 (고객의 나이, 거래 가격, 지역)  

>예시) 고객 데이터 (나이, 성별, 계정 생성일, 구매빈도)  
        종양 데이터 (크기나 모양, 색상의 진하기 등) - 흑백이미지

머신러닝에서는 **하나의 개체(행)**을 **샘플 sample** 또는 **데이터 포인트 data point** 라고 부른다.  
**샘플의 속성(열)**을 **특성 feature**이라고 한다. 

후에 좋은 입력 데이터를 만들어내는 특성 추출 feature extraction 혹은 특성 공학 feature engineering 도 다룰 것이다.

### 문제와 데이터 이해
데이터를 이해하고, 데이터로 해결할 문제와의 연관을 이해하는 것이 가장 중요! 다음과 같은 질문을 상기하자.

* 어떤 질문에 대한 답을 찾는가? 확보 데이터가 원하는 답을 줄 수 있는가?
* 질문을 머신러닝의 문제로 가장 잘 기술하는 방법은?
* 충분한 데이터를 수집했는가?
* 추출한 데이터의 특성은 무엇이고, 좋은 예측을 만들 수 있는가?
* 머신러닝 어플의 성과를 어떻게 측정할 수 있나?
* 머신러닝 솔루션이 다른 연구나 제품과 어떻게 협력할 수 있는가?

결국 문제를 푸는 건 사람. 전체 시스템에 대한 큰 그림을 가지고 있어야 한다.

## 왜 파이썬?
파이썬 python은 데이터 과학의 표준 프로그래밍 언어처럼 여겨진다. 

그 이유는?
* 범용 프로그래밍 언어의 장점 + 매트랩 MATLAB 과 R 같은 특정 분야를 위한 스크립팅 언어의 편리함
* 데이터 적재, 시각화, 통계, 자연어 처리, 이미지 처리 등에 필요한 라이브러리
* 터미널이나 주피터 노트북 jupyter notebook 같은 도구로 대화하듯 프로그래밍 가능
* 머신러닝과 데이터 분석을 위한 반복작업을 빠르고 손쉽게 처리 가능
* 복잡한 그래픽 사용자 인터페이스나 웹 서비스도 만들 수 있음
* 기존 시스템과의 통합이 용이함

## scikit-learn
오픈 소스 scikit-learn 사이킷런: 자유롭게 배포가능, 누구나 소스코드를 보고 실제로 어떻게 동작하는지 쉽게 확인 가능

### 설치
scikit-learn은 두 개의 다른 파이썬 패키지 NumPy 넘파이 와 SciPy 사이파이 를 사용한다. 그래프를 그리기 위해 matplotlib 맷플롯립, 대화식 개발을 위해 IPython 아이파이썬 과 주피터 노트북이 필요하다.  
필요한 패키지를 모아놓은 파이썬 배포판 설치:  
* Anaconda (https://www.continuum.io/anaconda-overview)  
* Enthought Canopy (https://www.enthought.com/products/canopy/)  
* Python(x,y) (https://python-xy.github.io/)  

이미 파이썬을 설치했다면 pip 명령을 이용해 필요한 패키지를 설치한다  
$ pip install numpy scipy matplotlib ipython scikit-learn pandas pillow

## 필수 라이브러리와 도구들
사이킷런을 구성하는 NumPy 와 SciPy와 pandas, matplotlib, mglearn에 대해서 알아보자.

### NumPy
Numpy 는 파이썬으로 과학 계산을 하려면 꼭 필요한 패키지이다. 다차원 배열을 위한 기능, 선형 대수 연산과 푸리에 변환 같은 고수준 수학 함수와 유사 난수 생성기를 포함한다.  
* 선형대수 : 벡터, 행렬, 선형 변환을 연구하는 수학 분야
* 푸리에 변환 : 시간의 함수인 신호 등을 주파수 성분으로 분해하는 변환  
* 유사 난수 : 초깃값을 이용하여 이미 결정되어 있는 메커니즘에 의해 생성되는 난수로, 초깃값을 알면 언제든 같은 값을 다시 만들 수 있으므로 진짜 난수와 구별하여 유사 난수라 한다.  

scikit-learn에서 NumPy 배열은 기본데이터 구조이다. scikit-learn은 NumPy 배열 형태의 데이터를 입력받는다. (우리가 사용할 데이터는 NumPy 배열로 변환되어야 함) 핵심기능은 다차원(*n*-차원) 배열인 ndarray 클래스입니다. 이 배열의 모든 원소는 동일한 데이터 타입이어야 한다. 


```python
import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6]])
print("x:\n{}".format(x))
```

    x:
    [[1 2 3]
     [4 5 6]]


### SciPy
Scipy는 과학 계산용 함수를 모아놓은 파이썬 패키지. 고성능 선형 대수, 함수 최적화, 신호 처리, 특수한 수학 함수와 통계 분포등의 기능 제공. scikit-learn은 알고리즘을 수현할 때 SciPy의 여러 함수를 사용하는데, 그 중 scipy.sparse가 가장 중요하다. 데이터 표현의 하나의 방법인 희소 행렬 기능을 제공한다. 희소행렬은 0을 많이 포함한 2차원 배열을 저장할 때 사용한다.


```python
from scipy import sparse

# 대각선 원소는 1이고 나머지는 0인 2차원 NumPy배열을 만든다.
eye = np.eye(4)
print("NumPy 배열:\n{}".format(eye))
```

    NumPy 배열:
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]]



```python
# NumPy 배열은 CSR 포맷(Compressed Row Storage의 약자, 행의 인덱스를 압축하여 저장)의 SciPy 희소 행렬로 변환
# 0이 아닌 원소만 저장
sparse_matrix = sparse.csr_matrix(eye)
print("Scipy의 CSR 행렬:\n{}".format(sparse_matrix))
```

    Scipy의 CSR 행렬:
      (0, 0)	1.0
      (1, 1)	1.0
      (2, 2)	1.0
      (3, 3)	1.0


보통 희소 행렬을 0을 모두 채운 2차원 배열로부터 만들지 않으므로 (메모리가 부족할 수 있어서) 희소 행렬을 직접 만들 수 있어야 한다. COO 포맷(Coordinate 포맷의 약자, 데이터가 놓일 행렬의 위치를 별도의 매개변수로 전달)


```python
data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print("COO 표현:\n{}".format(eye_coo))
```

    COO 표현:
      (0, 0)	1.0
      (1, 1)	1.0
      (2, 2)	1.0
      (3, 3)	1.0


### matplotlib
matplotlib 은 파이썬 대표 과학 계산용 그래프 라이브러리. **선 그래프, 히스토그램, 산점도** 등을 지원. 데이터와 분석결과를 다양한 관점에서 시각화해보면 통찰을 얻을 수 있다.  
jupyter notebook에서 사용할 땐, %matplotlib notebook 이나 %matplotlib inline 명령을 사용해 브라우저에서 바로 이미지를 확인 가능


```python
%matplotlib inline
import matplotlib.pyplot as plt

# -10에서 10까지 100개의 간격으로 나뉜 배열을 생성
x = np.linspace(-10, 10, 100)
# 사인 함수를 사용하여 y배열을 생성
y = np.sin(x)
# plot함수는 한 배열의 값을 다른 배열에 대응해서 선 그래프를 그린다.
plt.plot(x, y, marker="x")
```




    [<matplotlib.lines.Line2D at 0x181dd7b5b70>]




![png](/assets/img/output_19_1.png)


### pandas
pandas는 데이터 처리와 분석을 위한 파이썬 라이브러리. R의 data.frame을 본떠서 설계한 DataFrame이라는 데이터 구조를 기반.  
엑셀의 스프레드시트와 비슷한 테이블 형태. pandas는 이 테이블을 수정하고 조작하는 다양한 기능을 제공.  
특히, SQL처럼 테이블에 쿼리나 조인을 수행 가능.   
전체 배열의 원소가 동일한 타입이어야 하는 NumPy와는 달리 각 열의 타입이 달라도 된다.  
SQL, 엑셀 파일, CSV 파일 등을 데이터베이스에서 읽어 들일 수 있는 것이 pandas가 제공하는 유용한 기능 중 하나. 


```python
import pandas as pd

# 회원 정보가 들어간 간단한 데이터셋을 생성
data = {"Name" : ["0=2", "you", "youngeun", "영은"],
       "Location" : ["Seoul", "Paris", "Suwon", "London"],
       "Age" : [1, 23, 15, 24]
       }

data_pandas = pd.DataFrame(data)
# IPython.display는 주피터 노트북에서 Dataframe을 미려하게 출력해 줌
display(data_pandas)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Location</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0=2</td>
      <td>Seoul</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>you</td>
      <td>Paris</td>
      <td>23</td>
    </tr>
    <tr>
      <th>2</th>
      <td>youngeun</td>
      <td>Suwon</td>
      <td>15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>영은</td>
      <td>London</td>
      <td>24</td>
    </tr>
  </tbody>
</table>
</div>



```python
# 테이블에 질의하는 방법
# Age 열의 값이 20 이상인 행을 고른다.
display(data_pandas[data_pandas.Age > 20])
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Location</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>you</td>
      <td>Paris</td>
      <td>23</td>
    </tr>
    <tr>
      <th>3</th>
      <td>영은</td>
      <td>London</td>
      <td>24</td>
    </tr>
  </tbody>
</table>
</div>


### mglearn
이 라이브러리는 그래프나 데이터 적재와 관련한 상세 코드를 일일이 쓰지 않아도 되게끔 만든 유틸리티 함수, 그림을 그리거나 필요한 데이터를 바로 불러들이기 위해 사용. pip install mglearn 명령으로 설치.


```python
!pip install mglearn
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
```

    Collecting mglearn
      Downloading https://files.pythonhosted.org/packages/fb/01/8d3630ecc767c9de96a9c46e055f2a3a5f9e14a47d3d0348a36a5005fe67/mglearn-0.1.7.tar.gz (540kB)
    Requirement already satisfied: numpy in c:\programdata\anaconda3\lib\site-packages (from mglearn) (1.16.2)
    Requirement already satisfied: matplotlib in c:\programdata\anaconda3\lib\site-packages (from mglearn) (3.0.3)
    Requirement already satisfied: scikit-learn in c:\programdata\anaconda3\lib\site-packages (from mglearn) (0.20.3)
    Requirement already satisfied: pandas in c:\programdata\anaconda3\lib\site-packages (from mglearn) (0.24.2)
    Requirement already satisfied: pillow in c:\programdata\anaconda3\lib\site-packages (from mglearn) (5.4.1)
    Requirement already satisfied: cycler in c:\programdata\anaconda3\lib\site-packages (from mglearn) (0.10.0)
    Requirement already satisfied: imageio in c:\programdata\anaconda3\lib\site-packages (from mglearn) (2.5.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in c:\programdata\anaconda3\lib\site-packages (from matplotlib->mglearn) (1.0.1)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in c:\programdata\anaconda3\lib\site-packages (from matplotlib->mglearn) (2.3.1)
    Requirement already satisfied: python-dateutil>=2.1 in c:\programdata\anaconda3\lib\site-packages (from matplotlib->mglearn) (2.8.0)
    Requirement already satisfied: scipy>=0.13.3 in c:\programdata\anaconda3\lib\site-packages (from scikit-learn->mglearn) (1.2.1)
    Requirement already satisfied: pytz>=2011k in c:\programdata\anaconda3\lib\site-packages (from pandas->mglearn) (2018.9)
    Requirement already satisfied: six in c:\programdata\anaconda3\lib\site-packages (from cycler->mglearn) (1.12.0)
    Requirement already satisfied: setuptools in c:\programdata\anaconda3\lib\site-packages (from kiwisolver>=1.0.1->matplotlib->mglearn) (40.8.0)
    Building wheels for collected packages: mglearn
      Building wheel for mglearn (setup.py): started
      Building wheel for mglearn (setup.py): finished with status 'done'
      Stored in directory: C:\Users\유영은\AppData\Local\pip\Cache\wheels\74\cf\8d\04f4932d15854a36726c6210763c7127e62de28f5c8ddfcf3b
    Successfully built mglearn
    Installing collected packages: mglearn
    Successfully installed mglearn-0.1.7


## 소프트웨어 버전


```python
import sys
sys.version #python 버전
```




    '3.7.3 (default, Mar 27 2019, 17:13:21) [MSC v.1915 64 bit (AMD64)]'




```python
pd.__version__
```




    '0.24.2'



## 첫 번째 모델 : 붓꽃 데이터
상황 : 발견한 붓꽃의 품종을 알고 싶어서, 붓꽃의 꽃잎과 꽃받침의 폭과 길이를 측정. 이미 가지고 있는 setosa, versicolor, virginica 종으로 분류한 붓꽃 데이터를 가지고 발견한 붓꽃이 어떤 종인지 구분하기.  
> 붓꽃의 품종을 정확히 분류한 데이터를 가지고 있으므로 지도학습에 속함. 즉 분류 classification 문제, 출력될 수 있는 값들을 클래스 class 라고 함.

데이터 포인트 하나에 대한 기대 출력은 꽃의 품종. 특정 데이터 포인트에 대한 출력, 즉 품종은 **레이블 label** 이라고 한다

### 데이터 적재
붓꽃 데이터셋을 load_iris() 함수를 이용하여 적재


```python
from sklearn.datasets import load_iris
iris_dataset = load_iris()
```

iris 객체는 파이썬의 딕셔너리와 유사한 bunch 클래스의 객체로, 키와 값으로 구성


```python
iris_dataset.keys()
```




    dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])




```python
print(iris_dataset['DESCR'][:193] + "\n...")
```

    .. _iris_dataset:
    
    Iris plants dataset
    --------------------
    
    **Data Set Characteristics:**
    
        :Number of Instances: 150 (50 in each of three classes)
        :Number of Attributes: 4 numeric, pre
    ...


target_names의 값은 예측하려는 붓꽃 품종의 이름을 문자열로 가지고 있다.


```python
print("타킷의 이름:{}".format(iris_dataset['target_names']))
```

    타킷의 이름:['setosa' 'versicolor' 'virginica']


feature_names의 값은 각 특성을 설명하는 문자열 리스트


```python
print("특성의 이름: \n{}".format(iris_dataset['feature_names']))
```

    특성의 이름: 
    ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']


실제 데이터는 target과 data 필드에 들어있다. data는 꽃잎의 길이와 폭, 꽃받침의 길이와 폭을 수치 값으로 가지고 있는 NumPy 배열이다.


```python
print("data의 타입: {}".format(type(iris_dataset['data'])))
```

    data의 타입: <class 'numpy.ndarray'>


data 배열의 행은 개개의 꽃이 되고, 열은 각 꽃에서 구한 네 개의 측정치다.


```python
print("data의 크기: {}".format(iris_dataset['data'].shape))
```

    data의 크기: (150, 4)


이 배열은 150개의 붓꽃 데이터를 가지고 있다. 머신러닝에서 각 아이템은 샘플이라 하고, 속성은 특성이라 부른다. 그러므로 data 배열의 크기는 샘플의 수에 특성의 수를 곱한 값이다.


```python
print(iris_dataset['data'][:5])  # 데이터의 첫 다섯행
```

    [[5.1 3.5 1.4 0.2]
     [4.9 3.  1.4 0.2]
     [4.7 3.2 1.3 0.2]
     [4.6 3.1 1.5 0.2]
     [5.  3.6 1.4 0.2]]


target 배열도 샘플 붓꽃의 품종을 담은 NumPy 배열


```python
type(iris_dataset['target'])
```




    numpy.ndarray



target의 각 원소가 붓꽃 하나에 해당하는 1차원 배열


```python
iris_dataset['target'].shape
```




    (150,)



붓꽃의 종류는 0에서 2까지의 정수


```python
iris_dataset['target']
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])



숫자의 의미는 iris['target_names'] 배열에서 확인 가능. 0은 setosa, 1은 versicolor, 2는 virginica

### 성과 측정 : 훈련 데이터와 테스트 데이터
머신러닝 모델을 만들고 성능을 측정해야 한다. 모델의 예측을 신뢰할 수 있는지 확인하는 과정.  
하지만 모델을 구축할 때 사용한 데이터는 평가 목적으로 사용 불가, 훈련 데이터를 모두 기억해서 정확하게 맞출 수 있기 때문이다. 이런 상황은 모델이 일반화 되지 못한 것.  
모델의 성능을 측정하기 위해 : 레이블을 알고 있는 새 데이터를 모델에 적용해봐야 한다. 이를 위해 레이블된 데이터를 두 그룹으로 나눈다. 하나는 훈련 데이터(훈련 세트)로써 모델을 구축하는 데, 다른 하나는 테스트 데이터(데이터 세트 또는 홀드아웃 세트)로써 모델의 작동을 측정하는 데 사용한다.  
scikit-learn은 데이터셋을 섞어서 나눠주는 train_test_split 함수를 제공! 전체 행 중 75%를 레이블 데이터와 함께 훈련 세트로 뽑는다. 나머지 25%는 레이블 데이터와 함께 테스트 세트가 된다.  
데이터는 대문자 X로 표시하고, 레이블은 소문자 y로 표기한다. 수학의 표기 방식을 따르지만 데이터는 2차원 배열(행렬)이므로 대문자 X를, 타깃은 1차원 배열(벡터)이므로 소문자 y를 사용한다.


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0 )
```

train_test_split 함수로 데이터를 나누기 전에 유사 난수 생성기를 사용해 데이터셋을 무작위로 섞어야 한다. 만약 섞지 않은 상태로 데이터셋 뒤쪽을 테스트 세트로 이용하면, 그 데이터의 레이블은 모두 2가 된다. (데이터 포인트가 레이블 순서대로 정렬되어 있기 때문에)  
이 함수를 여러 번 실행해도 결과가 똑같이 나오도록 유사 난수 생성기에 넣을 난수 초깃값을 random_state 매개변수로 전달한다. 이렇게 하면 항상 같은 결과를 출력할 수 있다.  


```python
print("X_train 크기: {}".format(X_train.shape))
print("y_train 크기: {}".format(y_train.shape))
```

    X_train 크기: (112, 4)
    y_train 크기: (112,)



```python
print("X_test 크기: {}".format(X_test.shape))
print("y_test 크기: {}".format(y_test.shape))
```

    X_test 크기: (38, 4)
    y_test 크기: (38,)


### 가장 먼저 할 일 : 데이터 살펴보기

머신러닝 모델을 만들기 이전에 살펴보아야 할 것들. 
1. 머신러닝 없이 풀 수 있는 문제는 아닌지, 필요한 정보가 모두 수집되었는지 데이터를 조사
2. 데이터를 탐색하며 비정상적인 값이나 특이한 값을 찾는다. (측정 기준이 다르다거나)
3. 시각화를 통해 데이터 조사를 진행. (산점도 등이 유용함)  
   ! 산점도 : 데이터의 한 특성을 x축에 놓고 다른 하나는 y축에 놓아 각 데이터 포인트를 하나의 점으로 나타내는 그래프. 2차원의 특성만 확인 가능.
        이를 극복할 수 있는 산점도 행렬이 있음. 특성의 수가 적다면 좋음! (각 특성을 모두 짝지어 만들기 때문)
        산점도 행렬은 모든 특성의 관계가 한 그래프에 나타나는 것이 아니기 때문에 각각의 나누어진 산점도 그래프에 나타나지 않는 중요 성질이 있을 수 있다.


```python
# X_train 데이터를 사용해서 데이터 프레임을 만든다.
# 열 이름은 iris_dataset.feature_names에 있는 문자열을 사용
iris_df = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# 데이터프레임을 사용해 y_train에 따라 색으로 구분된 산점도 행렬을 만든다.
pd.plotting.scatter_matrix(iris_df, c=y_train, figsize=(15,15), marker ='o',
                           hist_kwds={'bins':20}, s=60, alpha=.8, cmap=mglearn.cm3)
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x00000181E13222B0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x00000181E110E470>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x00000181E1245710>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x00000181E12B1940>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x00000181E12DCBA8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x00000181E1384E10>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x00000181E13B60B8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x00000181E13DF358>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x00000181E13DF390>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x00000181E14707F0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x00000181E1497A58>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x00000181E14C2CC0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x00000181E14EBF28>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x00000181E151C1D0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x00000181E1546438>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x00000181E156E6A0>]],
          dtype=object)




![png](/assets/img/output_58_1.png)


### 첫 번째 머신러닝 모델 : k-최근접 이웃 알고리즘
머신러닝 모델 만들기 시작. **이해가 쉬운** k-최근접 이웃 k-Nearest Neighbors 분류기를 사용. 이 모델은 단순히 훈련 데이터를 저장하여 만든다. 새로운 데이터 포인트에 대한 예측이 필요하면, 알고리즘은 새 데이터 포인트에서 **가장 가까운 훈련 데이터 포인트를 찾는다**. 그 다음 찾은 훈련 데이터의 레이블을 새 데이터 포인트의 레이블로 지정한다.  
k-최근접 이웃 알고리즘에서 k는 가장 가까운 이웃 하나가 아니라 훈련 데이터에서 **새로운 데이터 포인트에 가장 가까운 k개의 이웃을 찾는다**는 뜻이다. 그런  다음 이 이웃들의 클래스 중 **빈도가 가장 높은 클래스를 예측값**으로 사용한다. 
사이킷런의 모든 머신러닝 모델은 Estimator라는 파이썬 클래스로 각각 구현되어 있다. KNN분류 알고리즘은 neighbors 모듈 아래 KNeighborsClassifier 클래스에 구현되어 있다. 모델을 사용하려면 클래스로부터 객체를 만들어야 한다. 이때 모델에 필요한 매개변수를 넣는다. KNeighborsClassifier에서 가장 중요한 **매개변수는 이웃의 개수**이다.  
지금은 하나의 이웃만 사용.



```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)
```

knn 객체는 훈련 데이터로 모델을 만들고 새로운 데이터 포인트에 대해 예측하는 알고리즘을 캡슐화한 것이다. 또한 알고리즘이 훈련 데이터로부터 추출한 정보를 담고 있다. KNeighborsClassifier의 경우는 훈련 데이터 자체를 저장하고 있다.  
훈련 데이터 셋으로부터 모델을 만드려면 knn 객체의 fit 메서드를 사용. 이 메서드는 훈련 데이터인 NumPy 배열 X_train과 훈련 데이터의 레이블을 담고 있는 NumPy 배열 y_train을 매개변수로 받는다.


```python
knn.fit(X_train, y_train)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=None, n_neighbors=1, p=2,
               weights='uniform')



fit 메서드는 knn 객체 자체를 반환하고 knn 객체 자체를 변경시킨다. 그래서 knn 객체가 문자열 형태로 출력된다. 이 출력에서 모델을 생성할 때 사용한 매개변수를 볼 수 있다. 거의 모든 매개변수가 기본값이고 n_neighbors=1은 직접 지정한 값이다. 사이킷런 모델들이 많은 매개 변수를 가지고 있지만 대부분은 성능을 최적화하거나 특별한 목적으로 사용한다. 출력 결과의 매개변수들을 신경 쓸 필요는 없음.

### 예측하기
예를 들어, 꽃받침의 길이가 5cm, 폭이 2.9cm, 꽃잎의 길이가 1cm, 폭이 0.2cm인 붓꽃을 보았을 때, 이 붓꽃의 품종은?


```python
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))
```

    X_new.shape: (1, 4)



```python
prediction = knn.predict(X_new)
print("예측: {}".format(prediction))
print("예측한 타깃의 이름:{}".format(iris_dataset['target_names'][prediction]))
```

    예측: [0]
    예측한 타깃의 이름:['setosa']


모델이 새로운 붓꽃의 품종을 클래스 0으로 예측했다. 이 결과값을 신뢰할 수 있을까? 

### 모델 평가하기
테스트 세트를 사용할 때가 왔다. 모델을 만들 때 사용하지 않았고 테스트 세트에 있는 각 붗꽃의 품종을 잘 알고 있는 데이터셋이다.
따라서 테스트 데이터에 있는 붓꽃의 품종을 예측하고 실제 레이블(품종)과 비교할 수 있다. 정확도를 계산해 모델의 성능을 평가한다.


```python
#테스트 셋에 대한 예측값
y_pred = knn.predict(X_test)
print(y_pred)
```

    [2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0 2 1 0 2 2 1 0
     2]



```python
#테스트 셋의 정확도
knn.score(X_test, y_test)
```




    0.9736842105263158



이 모델의 정확도는 약 0.97 으로, 테스트 셋에 포함된 붓꽃 중 97%를 정확히 맞혔다는 의미이다. 이 모델은 새로운 붓꽃에 대한 정확도가 97%일 것이라 기대할 수 있다. 정확도가 높으므로 아마추어 식물학자는 이 애플리케이션을 충분히 신뢰하고 사용할 수 있따. 이후의 장들에서 모델의 성능을 높이는 방법과 모델을 튜닝할 때의 주의점을 살펴보겠다.

## 요약, 정리
모델링은 세 개의 품종, 즉 세개의 클래스를 분류하는 문제였다. 분류 문제에서는 각 품종을 클래스라고 하며 개별 붓꽃의 품종은 레이블이라 한다.  
붗꽃 데이터셋은 두 개의 NumPy 배열로 이루어졌다. 하나는 데이터를 담고 있으며 사이킷런에서는 X로 표기한다. 다른 하나는 정확한 혹은 기대하는 출력을 가지고 있으며, y로 표기한다. 배열 X는 특성들의 2차원 배열이므로 각 데이터 포인트는 행 하나로 나타나고, 각 특성은 열 하나가 된다. 배열 y는 1차원 배열로 각 샘플의 클래스 레이블에 해당하는 0에서 2사이의 정수를 담고 있다.

이 데이터셋을 모델 구축에 사용할 훈련 세트와 모델이 새로운 데이터에 얼마나 잘 적용되는지 평가하기 위한 테스트 세트로 나눴다.

KNN분류 알고리즘은 새 데이터 포인트를 예측하기 위해 훈련 데이터에서 가장 가까운 이웃을 선택한다. 이 알고리즘은 KNeighborsClassifier 클래스에 구현되어 있으며 모델을 만드는 것은 물론 예측하는 기능도 제공한다. n_neighbors 매개 변수를 지정해 이 클래스의 객체를 만들었다. 그리고 훈련 데이터와 훈련 데이터의 레이블을 매개변수로 하여 fit 메서드를 호출해 모델을 만들었다. 정확도를 계산하는 score 메서드로 모델을 평가한다. 테스트 데이터와 테스트 데이터의 레이블을 score 메서드에 넣어 97% 정확도를 얻었다. 이는 테스트 세트에 있는 샘플의 97%를 정확히 맞혔다는 의미이다.

이 결과는 새로운 데이터에 이 모델을 적용해도 좋다는 확신을 주며, 대략 97% 확률로 이 모델을 신뢰할 수 있다.
