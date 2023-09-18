### ML Assignment #4


#### Execution description:

datapoints.py : 輸出dataset.txt
執行使用makefile 輸入: make data

linear model y = 2x +ε with zero-mean Gaussian noise ε∼ N(0, 1) to generate 500 data points with (equal spacing) x ∈ [−100, 100] 

train\_test.py: 將dataset.txt作為輸入資料集 
執行使用makefile 輸入: make train

先將dataset分為350筆訓練資料、150筆測試資料（subset.py），使用grid.py進行5-fold cross-validation 尋找best c and gamma 開始進行訓練（svm-train），手動輸入c and gamma參數，訓練使用RBF kernel，輸出模型後使用（svm-predict）將100筆測試資料用模型測試輸出準確率

scale\_train\_test.py: 將dataset.txt作為輸入資料集 
執行使用makefile 輸入: make scale
先將dataset.txt使用svm-scale進行縮放（-l -1 -u 1），使用grid.py進行5-fold cross-validation 尋找best c and gamma 開始進行訓練（svm-train），手動輸入c and gamma參數，訓練使用RBF kernel，輸出模型後使用（svm-predict）將所有資料用模型測試輸出準確率

#### Experimental results:

![](Pic/Aspose.Words.926edfe9-9b07-4d87-9f8f-51e6995f72e2.001.png)c = 32768  gamma = 0.00122

![](Pic/Aspose.Words.926edfe9-9b07-4d87-9f8f-51e6995f72e2.002.png)c = 32768  gamma = 0.000030517

![](Pic/Aspose.Words.926edfe9-9b07-4d87-9f8f-51e6995f72e2.003.png)c = 32768  gamma = 0.001

利用5-fold cross validation，至少找三組C and γ 參數

最好的參數 c = 32768  gamma = 0.000030517




在scaling之後使用未scaling的C=32768 and γ=0.000030517

![](Pic/Aspose.Words.926edfe9-9b07-4d87-9f8f-51e6995f72e2.004.png) 準確率較低 : 71.4%

在scaling之後進行5-fold cross-validation 尋找到的best C and γ 準確率很高:

99% 比尚未使用![](Pic/Aspose.Words.926edfe9-9b07-4d87-9f8f-51e6995f72e2.005.png)scaling的還好

#### Conclusion:

使用linear support vector machine時先將資料進行縮放（scaling）之後再進行模型訓練成效較好，但如果未進行縮放就訓練模型得到的參數並不適用縮放之後的資料集，會使得效果變差，這次經由調用LIBSVM的函數進行訓練，比較未縮放與縮放過後的資料集，比較的結論是先進行縮放再進行訓練可以獲得更高的準確性。

#### Discussion:

LIBSVM可以使用library也能進行訓練，但觀察easy.py這項檔案可以發現直接調用三項程式svm-train, svm-scale, svm-predict。
