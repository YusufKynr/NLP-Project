# -*- coding: utf-8 -*-
"""
Created on Wed May  1 17:42:31 2024

@author: yusuf
"""


import numpy as np
import pandas as pd
import pandas as pd
import keras
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras_preprocessing.text import Tokenizer








yorumlar = pd.read_csv('veri_seti.csv', sep=',', header=None, names=['sonuç', 'yorum'])

"noktalama işaretleri,sembolleri sil"
import re 


yorum = ""
sonuçlar = [] # olumsuz kelime hariç, verilerin son halidir
stopwords = [] # türkçe stopwordsler
veriler = [] # verilerin son halidir (olumsuz kelimeler dahil)
çıkartılan = [] # olumsuz kelimelerin eklerini çıkartılmasını engellemek için

"türkçe stopwordsleri dosyadan okuyup arraye aktar"
with open("stopwords.txt", 'r', encoding='utf-8') as dosya:
            for satir in dosya:
                stopwords.append(satir.strip())  # Her satırı diziye ekle, strip() ile gereksiz boşlukları temizle


"verileri küçültme ve kelime olarak split et"
def veri(yorum,i):
    "büyük-küçük harf problemi: hepsini küçült" 
    yorum = yorum.lower()

    "yorumu listeye çevir"
    yorum = yorum.split()
    


"stopwords'den arındır"      
def removeStopwords(yorum):
    yorum_list = yorum.split()  # Split the string into a list of words
    index = 0
    while index < len(yorum_list):
        kelime = yorum_list[index]
        if kelime in stopwords:
            yorum_list.pop(index)
        else:
            index += 1
    return ' '.join(yorum_list)  # Join the list back into a string and return





"olumsuz eki çıkartmama"
def removeNegativeWord(yorum):
    index = 0
    while index < len(yorum):
        kelime = yorum[index]
        if "sız" in kelime or "siz" in kelime or "suz" in kelime or "süz" in kelime:
            çıkartılan.append(yorum[index])
            yorum.remove(yorum[index])
        else:
            index += 1
           
        
 

"gövde ve eki ayrıştırma işlemi"
from zeyrek import MorphAnalyzer
zeyrek = MorphAnalyzer()
def stemmer(yorum):
    kelimeler = yorum.split()  # Stringi kelimelere ayır
    for kelime in kelimeler:  # Her bir kelime için döngüyü çalıştır
        sonuç = zeyrek.lemmatize(kelime)  # Her kelimenin kökünü bul
        sonuçlar.append(min(sonuç[0][1], key=len).lower())  # En kısa kökü seç ve results listesine ekle



def main():
    for i in range(len(yorumlar)):
        yorum = re.sub('[^a-zA-ZçğıöşüÇĞİÖŞÜ]',' ', yorumlar["yorum"][i])
        veri(yorum,i)
        removeStopwords(yorum)
        removeNegativeWord(yorum) 
        stemmer(yorum)
        sonSonuç = sonuçlar + çıkartılan
        sonSonuç = ' '.join(sonSonuç)
        veriler.append(sonSonuç)
        çıkartılan.clear()
        sonuçlar.clear()


main()

"Vektör sayacı"
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=(2000))
X = cv.fit_transform(veriler).toarray() #bağımsız değişken
y = yorumlar["sonuç"].values

"Makine Öğrenmesi"
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = None)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)

"Hata matrixi hesaplama"
y_predict = gnb.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)

print("Naive Bayes Doğruluk:", (cm[0,0] + cm[1,1]) / np.sum(cm) *100)
print(cm) #hata matrisi




# Veri setini yükleme
yorumlar = pd.read_csv('veri_seti.csv', sep=',', header=None, names=['sonuç', 'yorum'])

# Metin ve etiketlerin ayrılması
X = yorumlar['yorum'].values
y = yorumlar['sonuç'].values

# Etiketleri sayısal değerlere dönüştürme
le = LabelEncoder()
y = le.fit_transform(y)

# Metin verisini sayısal vektörlere dönüştürme
max_words = 1000
max_len = 150
tokenizer = Tokenizer(num_words=max_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True)
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
X = tf.keras.utils.pad_sequences(sequences, maxlen=max_len)

# Veri setini eğitim ve test setlerine böleme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LSTM tabanlı derin öğrenme modeli oluşturma
embedding_dim = 500

model = keras.Sequential()
model.add(keras.layers.Embedding(max_words, embedding_dim, input_length=X.shape[1]))
model.add(keras.layers.SpatialDropout1D(0.2))
model.add(keras.layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Modeli eğitme
batch_size = 32
epochs = 6
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=2)

# Modeli değerlendirme
score = model.evaluate(X_test, y_test, verbose=0)
print("Test Loss:", score[0])
print("Test Accuracy:", score[1])



