### ML Assignment #6

#### Execution description:

執行方式於 dataset在同一目錄下執行 python ML\_hw6.py

藉由在data\_transforms 內加入transforms 方法擴增資料集

#### Experimental results:

1\. Weak Augmentation挑選一種data transforms的方法，比較只使用原資料集 vs. 增強後資料集，模型準確率的差異

原資料集執行結果：

![](Pic/Aspose.Words.929c91b6-62fb-45b5-89c3-35e43a55d01f.003.png)

![](Pic/Aspose.Words.929c91b6-62fb-45b5-89c3-35e43a55d01f.001.png)

![](Pic/Aspose.Words.929c91b6-62fb-45b5-89c3-35e43a55d01f.002.png)

transforms.RandomAffine(32, translate=None, scale=[1,2], shear=4)

保持像素的分布中心不變，對圖片做隨機放射變換

執行結果：（原資料＋擴增資料集＝原資料集2倍）

![](Pic/Aspose.Words.929c91b6-62fb-45b5-89c3-35e43a55d01f.004.png)

![](Pic/Aspose.Words.929c91b6-62fb-45b5-89c3-35e43a55d01f.005.png)

![](Pic/Aspose.Words.929c91b6-62fb-45b5-89c3-35e43a55d01f.006.png)

transforms.RandomHorizontalFlip(0.5)

以給定的機率隨機水平翻折圖片
執行結果：（原資料＋擴增資料集＝原資料集2倍）


![](Pic/Aspose.Words.929c91b6-62fb-45b5-89c3-35e43a55d01f.007.png)

![](Pic/Aspose.Words.929c91b6-62fb-45b5-89c3-35e43a55d01f.009.png)

![](Pic/Aspose.Words.929c91b6-62fb-45b5-89c3-35e43a55d01f.008.png)


transforms.AugMix(severity= 8,mixture\_width=4)
對圖片進行不同的數據增強（Aug），然後混合（Mix）多個數據增強後的圖片; 執行結果：（原資料＋擴增資料集＝原資料集2倍）

![](Pic/Aspose.Words.929c91b6-62fb-45b5-89c3-35e43a55d01f.012.png)

![](Pic/Aspose.Words.929c91b6-62fb-45b5-89c3-35e43a55d01f.010.png)

![](Pic/Aspose.Words.929c91b6-62fb-45b5-89c3-35e43a55d01f.011.png)

多種data transforms的資料集

（原資料集＋RandomHorizontalFlipy資料集＋RandomAffine資料集＋AugMix資料集=原資料集4倍）執行結果：

![](Pic/Aspose.Words.929c91b6-62fb-45b5-89c3-35e43a55d01f.013.png)

2\. Strong Augmentation使用4~6種data transforms，同時作用於原始資料集，比較只使用原資料集 vs. 增強後資料集，模型準確率的差異

4種datatransforms :

(1)transforms.RandomHorizontalFlip(0.5)

(2)transforms.RandomAffine(32, translate=None, scale=[1,2], shear=4)

(3)transforms.AugMix(severity= 8,mixture\_width=4)

(4)transforms.RandomGrayscale(0.1)

執行結果：（原資料＋擴增資料集＝原資料集2倍）

![](Pic/Aspose.Words.929c91b6-62fb-45b5-89c3-35e43a55d01f.016.png)

![](Pic/Aspose.Words.929c91b6-62fb-45b5-89c3-35e43a55d01f.014.png)

![](Pic/Aspose.Words.929c91b6-62fb-45b5-89c3-35e43a55d01f.015.png)


將資料集增加（原資料＋擴增資料集\*3＝原資料集4倍）


![](Pic/Aspose.Words.929c91b6-62fb-45b5-89c3-35e43a55d01f.017.png)

#### Conclusion:

將原資料集加上弱增強的資料集，使資料集變為兩倍，實驗中都有比只有原資料集準確率更高，串連三種弱增強資料集，使得總體資料量是原資料的4倍時，結果比強增強資料集（問題二）的結果更好，可能的原因是資料量較大 ; 問題二強增強的資料集，用四種data transforms作用在原資料集再加上原資料集，使資料集變為兩倍，成果比單一弱增強資料集準確率高但並沒有顯著的提升，可能在參數調整之後，弱增強會比強增強資料集表現更好，**而在相同參數的結果下強增強資料集比單一弱增強資料集好，但比多個弱增強資料集的結果差，當強增強資料量與多個弱增強資料集資料量相同時（實驗中都是原資料集的四倍），結果顯示強擴增資料集的準確率較高**，而在未使用兩倍的資料集前，嘗試直接將弱增強資料集進入模型訓練，其結果準確率皆比原資料集差：

使用RandomAffine （擴增資料集 與原資料集大小相同）

![](Pic/Aspose.Words.929c91b6-62fb-45b5-89c3-35e43a55d01f.018.png)

準確率比原資料集更差

#### Discussion:

一開始都只使用資增強資料集，並沒有加上原資料集，結果準確率都比較差，觀察data size之後，利用torch.utils.data 內的ConcatDataset將資料集做連結獲得兩倍的資料集，準確率就有所提升，而起初遇到concat的資料集並沒有.class type，但因為資料集的class都相同，所以可以用original的image folder 的.class即可。
