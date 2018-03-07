# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 14:22:13 2018

@author: k
"""



import numpy as np 
import pandas as pd 
import keras
import cv2
import sys
from keras.models import Model
from keras.layers import Dense, Flatten
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.applications.vgg19 import VGG19


if len(sys.argv) != 4:
	print ("Usage: python dog_breed.py [im_size] [model_epoch] [model_name]")
	exit()

im_size,  model_epoch = int(sys.argv[1]), int(sys.argv[2])
model_name = sys.argv[3]

x_train = []
y_train = []
x_test = []


df_train = pd.read_csv('/dog/label/labels.csv')
df_test = pd.read_csv('/dog/sample_submission/sample_submission.csv')

#one-hot encoding
targets_series = pd.Series(df_train['breed'])
one_hot = pd.get_dummies(targets_series, sparse = True)
one_hot_labels = np.asarray(one_hot)


#train 데이터 불러오기
print("train data loading..")
i = 0 
for f, breed in tqdm(df_train.values):
    img = cv2.imread('/dog/train/train/{}.jpg'.format(f))
    label = one_hot_labels[i]
    x_train.append(cv2.resize(img, (im_size, im_size)))
    y_train.append(label)
    i += 1
    
#test데이터 불러오기
print("test data loading..")
for f in tqdm(df_test['id'].values):
    img = cv2.imread('/dog/test/test/{}.jpg'.format(f))
    x_test.append(cv2.resize(img, (im_size, im_size)))

y_train_raw = np.array(y_train, np.uint8) 
x_train_raw = np.array(x_train, np.float32) / 255.
x_test  = np.array(x_test, np.float32) / 255. 


#breed 개수, 총 120종이 있다
num_class = y_train_raw.shape[1]

#train데이터를 7:3으로 나눈다
X_train, X_valid, Y_train, Y_valid = train_test_split(x_train_raw, y_train_raw, test_size=0.3, random_state=1)


if model_name == "VGG19":
    base_model = VGG19(weights = 'imagenet', include_top=False, input_shape=(im_size, im_size, 3))

#모델 선택
if model_name == "VGG19":
    base_model = VGG19(weights = 'imagenet', include_top=False, input_shape=(im_size, im_size, 3))
elif model_name == "ResNet50":
    from keras.applications.resnet50 import ResNet50
    base_model = ResNet50(weights = 'imagenet', include_top=False, input_shape=(im_size, im_size, 3))
elif model_name == "Xception":
    from keras.applications.xception import Xception
    base_model = Xception(weights = 'imagenet', include_top=False, input_shape=(im_size, im_size, 3))
elif model_name == "InceptionV3":
    from keras.applications.inception_v3 import InceptionV3
    base_model = InceptionV3(weights = 'imagenet', include_top=False, input_shape=(im_size, im_size, 3))
elif model_name =="InceptionResNetV2":
    from keras.applications import InceptionResNetV2
    base_model = InceptionResNetV2(weights = 'imagenet', include_top=False, input_shape=(im_size, im_size, 3))
else:
    print("you got wrong model, base model will be VGG19")
    base_model = VGG19(weights = 'imagenet', include_top=False, input_shape=(im_size, im_size, 3))


#기존 vgg19 아웃풋을 평평하게 편다음에 우리가 만든 output출력층을 붙여준다
x = base_model.output #기존 아웃풋을
x = Flatten()(x) #평평하게 편다음에
predictions = Dense(num_class, activation='softmax')(x) #우리가 만든 output출력층을 
model = Model(input=base_model.input, output=predictions) #붙여준다 

for layer in base_model.layers:
    layer.trainable = False

model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])


#모델을 훈련시킨다
model.fit(X_train, Y_train, nb_epoch=model_epoch, validation_data=(X_valid, Y_valid), verbose=1)

#데이터 넣으면 예상값 반환
preds = model.predict(x_test, verbose=1) #verbose = 1 로깅을 프로그레스 바로 보여준다
#결과값 보기좋게 정리
sub = pd.DataFrame(preds)
#원핫 인코딩으로 이름변경
col_names = one_hot.columns.values
sub.columns = col_names
#id값 넣기
sub.insert(0, 'id', df_test['id'])
#결과값 csv로 출력
sub.to_csv('result.csv', encoding='utf-8',index=False)

#모델저장
json_string = model.to_json()
model.save('result_model.h5')

model.to_json()
import json
with open('result.txt','w') as outfile:
    json.dump(model.to_json(),outfile)


"""
#모델 재사용
import json
from keras.models import model_from_json

with open('result.txt') as json_data:
    d = json.load(json_data)
    
model = model_from_json(d)
model.load_weights('result_model.h5')
"""
