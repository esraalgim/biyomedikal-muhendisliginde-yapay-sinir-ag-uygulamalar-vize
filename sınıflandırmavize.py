# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 21:35:31 2020

@author: Esra
"""
# pandas, numpy ve matplotlib kütüphaneleri tanımlandı.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#veri seti tanımlandı.
veriseti=pd.read_csv("column_2C_weka.csv")
print(veriseti.head(310))

#veri setinde kayıp ya da sayısal olmayan verilerin olup olmadığının kontrolü için bu kod bloğu oluşturuldu.
import re
kayip_veriler=[]
sayisal_olmayan_veriler=[]

for oznitelik in veriseti:
    essiz_deger=veriseti[oznitelik].unique()
    print("'{}' özniteliğine ait (unique) veriler : {}".format(oznitelik, essiz_deger.size))
    if (essiz_deger.size>10):
        print("10 adet essizdeger listele")
        print(essiz_deger[0:10])
        print("\n---------\n")
        
        if(True in pd.isnull(essiz_deger)):
            s="{} özniteliğe ait kayıp veriler {}".format(oznitelik, pd.isnull(veriseti[oznitelik]).sum())
            kayip_veriler.append(s)
            
            for i in range (1,np.prod(essiz_deger.shape)):
                if (re.match('nan',str(essiz_deger[i]))):
                    break
                if not (re.search('(^\d+\.?\d*$)|(^\d*\.?\d+$)', str(essiz_deger[i]))):
                    sayisal_olmayan_veriler.append(oznitelik)
                    break
                print("Kayıp veriye sahip öznitelikler:\n{}\n\n".format(kayip_veriler))
                print("Sayısal olmayan veriye sahip öznitelikler:\n{}".format(sayisal_olmayan_veriler))
                
#%%
                
Abnormal=veriseti[veriseti.clas=="Abnormal"]
Normal=veriseti[veriseti.clas=="Normal"]

veriseti.clas=[1 if each=="Abnormal" else 0 for each in veriseti.clas]
y=veriseti.clas.values
x_veriseti=veriseti.drop(["clas"],axis=1)       


#%% veri setinin eğitim ve  test kümelerine ayrılması ve öznitelik ölçeklendirme için uygulanan kod bloğu.
          
from sklearn.model_selection import train_test_split
x=veriseti.iloc[:,:-1].values
y=veriseti.iloc[:,-1].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3, random_state=1)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train) 
x_test=scaler.fit_transform(x_test)
y_train=scaler.fit_transform(y_train.reshape(-1,1)).ravel() 
y_test=scaler.fit_transform(y_test.reshape(-1,1)).ravel()            
                
#%% eğitim setinin KNN algoritmasına göre uydurulması ve elde edilen test sonucu tahminleri

from  sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn.fit(x_train,y_train)
prediction=knn.predict(x_test)
print("knn score:", knn.score(x_test,y_test))

# test sonucunun tahmin edilmesi
y_pred = knn.predict(x_test)
                

#%%

score_list=[]

for each in range (1,30):
    knn2=KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
    
plt.plot(range(1,30),score_list)
plt.xlabel("k")
plt.ylabel("accuracy")
plt.show
       

#%%

from sklearn.svm import SVC
svm=SVC(random_state=1,gamma="auto")
svm.fit(x_train,y_train)
print("svm accuracy:",svm.score(x_test,y_test))

#%% rf classification

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=100,random_state=1)
rf.fit(x_train,y_train)

print("rf accuracy:",rf.score(x_test,y_test))
 #%% hata matrisi oluşturulmuş ve performans değerlendirme ölçütleri listelenmiştir.

from sklearn.metrics import confusion_matrix, classification_report
hm=confusion_matrix(y_test, y_pred)
print(classification_report(y_test,y_pred))

#%%
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
ypo,dpo, esikDeger=roc_curve(y_test, y_pred)
aucDegeri =auc(ypo,dpo)

plt.figure()
plt.plot(ypo,dpo, label='AUC %0.2f' % aucDegeri)
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('yanlış pozitif oranı(YPO)')
plt.ylabel('dogru pozitif oranı(DPO)')
plt.title('roc egrisi')
plt.legend(loc="best")
plt.show()














