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
sonuçlar = []
stopwords = []

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
çıkartılan = []
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




"Vektör sayacı"
#from sklearn.feature_extraction.text import CountVectorizer
#cv = CountVectorizer(max_features=(2000))
#X = cv.fit_transform(results)
veriler= []
def main():
    for i in range(len(yorumlar)):
        yorum = re.sub('[^a-zA-ZçğıöşüÇĞİÖŞÜ]',' ', yorumlar["yorum"][i])
        veri(yorum,i)
        stemmer(yorum)
        removeStopwords(yorum)
        removeNegativeWord(yorum) 
        sonSonuç= sonuçlar + çıkartılan
        veriler.append(sonSonuç)
        çıkartılan.clear()
        sonuçlar.clear()


main()
print(veriler)