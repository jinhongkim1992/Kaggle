from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from datetime import datetime
import numpy as np

import pandas as pd

#데이터 추출
X = pd.read_csv("C:/Users/USER/Desktop/파이썬/Kaggle/Titanic/train.csv")
X_test = pd.read_csv("C:/Users/USER/Desktop/파이썬/Kaggle/Titanic/test.csv")
submission = pd.read_csv("C:/Users/USER/Desktop/파이썬/Kaggle/Titanic/gender_submission.csv")
Y = X.pop('Survived')

#데이터 전처리를 위한 train 데이터와 test데이터 병합
All_X = pd.concat([X, X_test], axis = 0).reset_index()
#전처리 이후 데이터 분리를 위한 데이터 index 추출
train_index = list(range(len(X)))
test_index = list(range(len(X), len(All_X))
#데이터 인덱스 추출값 확인 (-1 이여야 정상)
max(train_index) - min(test_index)

#나이 평균값으로 null 값 채워줌
All_X.isnull().sum()
All_X['Age'].fillna(All_X.Age.mean(),inplace = True)
All_X.isnull().sum()
All_X.dtypes
#Pclass는 범주형 데이터로 변경
All_X['Pclass'] = All_X['Pclass'].astype('object')
All_X.drop(['PassengerId', 'Name', 'Ticket'], axis = 1, inplace = True)

#범주형 데이터에 대해서는 one hot encoding 처리
object_variables = list(All_X.dtypes[All_X.dtypes == 'object'].index)
object_variables

for variables in object_variables:
    temp = pd.get_dummies(All_X[variables], prefix = variables)
    All_X = pd.concat([All_X, temp], axis = 1)
    All_X.drop([variables], axis = 1, inplace = True)

All_X.dtypes

#one hot encoding된 데이터들은 범주형 데이터로 설정
object_variables = list(All_X.dtypes[All_X.dtypes == 'uint8'].index)
object_variables
for var in object_variables:
    All_X[var] = All_X[var].astype('object')
#필요없는 column 삭제
del All_X["index"]
#위 train , test 데이터 구분을 위한 index값을 토대로 전처리 완료된 데이터 내용 분리
train_X = All_X.iloc[train_index]
test_X = All_X.iloc[test_index]
train_X.isnull().sum()
#테스트 데이터에서 fare에 대한 Nan 값 평균값으로 처리
test_X['Fare'].fillna(test_X.Fare.mean(), inplace = True)

#랜덤포레스트 모델링 구축을 위해서 하이퍼 파라미터값 조합 설정
es = [100,200,300,400,500,600,700,800,900,1000,1500,2000,2500]
max_leaf = list(range(5,30))
max_leaf
empty_list = []

for i in es:
    for j in max_leaf:
        model = RandomForestClassifier(n_estimators = i,
                                       oob_score = True,
                                       max_leaf_nodes = j)

        a = datetime.now()
        model.fit(train_X, Y)
        score = model.oob_score_
        print(score)
        empty_list.append([i,j,score])
        print(datetime.now() - a)

copy_list = empty_list

copy_list = np.array(copy_list)
copy_list = copy_list[np.argsort(-copy_list[:,2])]
a = int(copy_list[1][1])
b = int(copy_list[1][0])

#validation 검증에서 가장 높은 score를 낸 하이퍼 파라미터로 모델 설정
model = RandomForestClassifier(n_estimators = b, oob_score = True, max_leaf_nodes = a)
model.fit(train_X, Y)
score = model.oob_score_
score

answer = model.predict(test_X)
pd.DataFrame(answer)

submission = pd.concat([submission, pd.DataFrame(answer)], axis = 1)

submission.drop(submission.columns[[1]],axis='columns',inplace = True)
submission.columns = ['PassengerId', 'Survived']
submission.to_csv("C:/Users/USER/Desktop/파이썬/Kaggle/Titanic/submission.csv", mode='w', index = False)
