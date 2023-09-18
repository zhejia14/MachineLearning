### ML Assignment #3

#### Problem 1 :

Adaboost parameters

![](Aspose.Words.241dadcd-53d4-4f91-9fb2-236cd761f69c.001.png)

train: 訓練資料集![](Aspose.Words.241dadcd-53d4-4f91-9fb2-236cd761f69c.002.png)

train\_label: 訓練資料的真實類別

test: 測試資料集

test\_label : 測試資料集的真實類別

cycles: 迭代次數（弱分類器數量）

distribution: 樣本的權重分佈，在一開始根據資料量設定權重皆為 1

![](Aspose.Words.241dadcd-53d4-4f91-9fb2-236cd761f69c.003.png)

#### Problem 2:

![](Aspose.Words.241dadcd-53d4-4f91-9fb2-236cd761f69c.004.png)

weakleaner: 弱學習器調用 weakLearner 函數

![](Aspose.Words.241dadcd-53d4-4f91-9fb2-236cd761f69c.005.png)

在 weakleaner 函數中 distribution 將 threshold 的區間值設定為 16，根據 feature 與 label的差值是不是大於或等於 16的倍數計算加權的錯誤率

![](Aspose.Words.241dadcd-53d4-4f91-9fb2-236cd761f69c.006.png)

找出錯誤率最小的弱分類器 0.5是隨機猜測的狀況，所以越接近的表示錯誤率越高

learning algorithm A 挑選計算錯誤率離0.5最遠的樣本與分類器來調整參數

![](Aspose.Words.241dadcd-53d4-4f91-9fb2-236cd761f69c.007.png)

之後更新 distribution : abs(label-(train(:,i)>=t))預測與真實的差異值，1 減去此差值使得分類錯誤的樣本獲得更的大的權重，正確樣本的權重減少，beta(j):權重 最後將 distribution 確保總和是 1

並沒有使用 bootstrapped，input 由 1~100,1~200…,1~1000 輸入弱分類器

#### Problem 3:

![](Aspose.Words.241dadcd-53d4-4f91-9fb2-236cd761f69c.008.png)

![](Aspose.Words.241dadcd-53d4-4f91-9fb2-236cd761f69c.009.png)

將 s( beta(j))、feature i (i)、threshold θ(t) 放入 boosted 陣列

$ℎ_{s,i,θ}^{1}= 0.3774 ∗sign(x_{11} − 80)$

weaker leaner 1:feature i =11 第 11個特徵分割樣本，θ=80 大於 θ 分配一 個標籤，小於等於 θ 分配另一個標籤，此decision stump 的權重 s 為 0.3774

$ℎ_{s,i,θ}^{2}= 0.4807 ∗sign(x_{170} − 80)$

weaker leaner 2:feature i =170 第 170個特徵分割樣本，θ=80 大於 θ 分配 一個標籤，小於等於 θ 分配另一個標籤，此 decision stump 的權重 s 為 0.4807

$ℎ_{s,i,θ}^{3}= 2.017 ∗sign(x_{58} − 16)$

weaker leaner 3:feature i =58 第 58個特徵分割樣本，θ=16 大於 θ 分配一 個標籤，小於等於 θ 分配另一個標籤，此 decision stump 的權重 s 為 2.017

#### Problem 4 :

$BlendingWeight\quad\alpha_{j}=ln(\frac{error}{1-error})=ln(beta(j))$

$\alpha_1=-0.97444$
$\alpha_2=−0.73251$
$\alpha_3=0.70161$
