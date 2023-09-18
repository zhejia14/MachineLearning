### ML Assignment #2
#### Execution description:
lineData.py (model one) sinData.py (model 2) 輸出 lineDataset.txt sinDataset.txt 
執行使用 makefile 輸入: make dataset

linearRegression.py 根據資料集進行 linear regression 
執行輸入(Example): python linearRegression.py inputfile.txt

Polynomial Regression:
根據資料集進行 polynomial regression 
Degree5_polynomialRegression.py 5 次方
Degree10_polynomialRegression.py 10 次方
Degree14_polynomialRegression.py 14 次方
執行輸入(Example): python Degree5_polynomialRegression.py inputfile.txt

dataPoint.py (model 2)根據輸入數字創造該數量資料點 輸出 points.txt
執行輸入(Example): python datapoint.py 320

regularization.py 正規化根據問題六 輸入 λ 值的分子 e.g. 0, 0.001, 1, 1000
執行輸入(Example): python regularization.py 1000
#### Experimental results: 各項實驗數據:experimental_result.pdf
#### Problem4 :
Compare the results with linear/polynomial regression on different datasets.
在資料集呈現偏向線性時 linear regression 可以找到一個合適的線性函數， polynomial regression 隨著 degree 增加 training error 下降當 degree=14 時過度擬 合 training error 極低 Five-fold cross-validation errors 極高，而當資料呈現 model 2 sin 函數類型時 linear regression 所顯示的圖表現並不好，polynomial regression 所顯示的圖與 sin 函數類似 training error 及 Five-fold cross-validation errors 的數值都不高在 degree=10、 degree=14 時隨次方提高會擬合數據，Five- fold cross-validation errors 數值越來越高
#### Problem5 :
Compare the results to those in 4).
在都為 sin 函數類型資料集，且都是 14 次方 polynomial regression 10 筆資料與 問題四的 15 筆資料結果類似 Five-fold cross-validation errors 的數值都偏高，可 能是數據不足，當資料數增加到 80、320 筆時 Five-fold cross-validation errors 有 明顯的下降
#### Problem6 :
隨著 λ 值變化在同樣類型的函數資料 Five-fold cross-validation errors 值可能會 下降，但如果模型複雜度太低也會導致數據變差 e.g. 1/m 1000/m
#### Conclusion:
linear regression 的目標是找到最佳擬合線，使得觀察到的資料點和預測值之間 的距離最小，polynomial regression 的目標是找到最佳擬合多項式函數來描述數 據，在這實驗中我們發現兩種迴歸方式所呈現的結果，主要的差異在於線性的 假設，為了避免模型過度擬合加入 regularization 的方式對訓練數據的擬合能力 進行抑制，降低驗證數據上的錯誤率，但若模型變得太簡單，無法很好地擬合 數據，在驗證數據上的表現也會變差，需要在適當的範圍內調整λ值，以獲得 最佳的模型性能，當 training error 與 Five-fold cross-validation errors 差距很大 時表示模型出現過度擬和的現象，而資料點過少也會造成數據不準確，在資料 集配合適用的迴歸方法才能得到好的模型結果
#### Discussion:
在計算 training error 時雖然有現有的 function 可以使用，但跑出來的數據卻非 常奇怪，所以自己寫公式帶入。
