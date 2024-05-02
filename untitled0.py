# -*- coding: utf-8 -*-
"""
Created on Wed May  1 17:42:31 2024

@author: yusuf
"""

import numpy as np
import pandas as pd
import nltk

yorumlar = pd.read_csv('veri_seti.csv')

"noktalama işaretleri,sembolleri sil"
import re 
yorum = re.sub('[^a-zA-ZçğıöşüÇĞİÖŞÜ]',' ', yorumlar['yorum'][68])

"büyük-küçük harf problemi: hepsini küçült" 
yorum = yorum.lower()

"yorumu listeye çevir"
yorum = yorum.split()

"türkçe stopwordsleri dosyadan okuyup arraye aktar"
stopwords = []
with open("stopwords.txt", 'r', encoding='utf-8') as dosya:
            for satir in dosya:
                stopwords.append(satir.strip())  # Her satırı diziye ekle, strip() ile gereksiz boşlukları temizle
                
"stopwords'den arındırma"      
def removeStopwords(array):
    index = 0
    while index < len(array):
        kelime = array[index]
        if kelime in stopwords:
            array.pop(index)
        else:
            index += 1

removeStopwords(yorum)

"gövde ve eki ayrıştırma işlemi"
from zeyrek import MorphAnalyzer
nltk.download('punkt')
zeyrek = MorphAnalyzer()

results = []

for index in range(len(yorum)):
    
    result = zeyrek.lemmatize(yorum[index].lower())
    results.append(min(result[0][1], key=len))  # En kısa elemanı seç ve results listesine ekle

print(results)


"bazı kelimelerin birden fazla potansiyel kökü olabiliyor, bunun için bir tanesini seçiyoruz"


