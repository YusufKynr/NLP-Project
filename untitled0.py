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
yorum = re.sub('[^a-zA-ZçğıöşüÇĞİÖŞÜ]',' ', yorumlar['yorum'][29])

"büyük-küçük harf problemi: hepsini küçült "
yorum = yorum.lower()

"yorumu listeye çevir"
yorum = yorum.split()

"türkçe stopwordsleri dosyadan okuyup arraye aktar"
nltk.download('stopwords')
from nltk.corpus import stopwords

#print(stopwords)

"gövde ve eki ayrıştırma işlemi"

from zeyrek import MorphAnalyzer

nltk.download('punkt')
zeyrek = MorphAnalyzer()
results = []
for index in range(len(yorum)):
    result = zeyrek.lemmatize(yorum[index])
    results.append(result[0])
print(results)
yorum = [zeyrek.lemmatize(kelime) for kelime in yorum if not kelime in set(stopwords.words('turkish'))]

