# -*- coding: utf-8 -*-
"""
Created on Wed May  1 17:42:31 2024

@author: yusuf
"""

import numpy as np
import pandas as pd


yorumlar = pd.read_csv('veri_seti.csv')

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)

"Hata matrixi hesaplama"
y_predict = gnb.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)

# Toplam örnek sayısını hesapla
toplam_ornek_sayisi = np.sum(cm)

# Hata matrisini yüzdelik olarak hesapla
yuzdelik_cm = cm / toplam_ornek_sayisi * 100

# Köşegenin toplamını hesapla (doğru tahminlerin toplamı)
dogru_tahminlar = np.sum(np.diag(cm))

# Toplam yüzdeyi hesapla
toplam_yuzde = dogru_tahminlar / toplam_ornek_sayisi * 100

print("Toplam doğru sınıflandırma yüzdesi:", toplam_yuzde)
print(cm)
