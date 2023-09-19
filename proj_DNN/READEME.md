### ML Assignment #5

#### Execution description:

依據檔名分別為：

Q1 : nblock\_1 , nblock\_10 , nblock\_50 (block數)

Q2 : inmodel\_block\_16 , inmodel\_block\_256 (model內block的層數)

使用kaggle執行程式碼

#### Experimental results:

1\. 更改model的block數
block : 1


![](Pic/Aspose.Words.6faf1351-b469-40d1-813b-aca201e8a447.001.png)

![](Pic/Aspose.Words.6faf1351-b469-40d1-813b-aca201e8a447.002.png)

block : 10 


![](Pic/Aspose.Words.6faf1351-b469-40d1-813b-aca201e8a447.003.png)


block : 50


![](Pic/Aspose.Words.6faf1351-b469-40d1-813b-aca201e8a447.004.png)

![](Pic/Aspose.Words.6faf1351-b469-40d1-813b-aca201e8a447.005.png)




2\.更改model內block的層數

更改16 


![](Pic/Aspose.Words.6faf1351-b469-40d1-813b-aca201e8a447.006.png)

![](Pic/Aspose.Words.6faf1351-b469-40d1-813b-aca201e8a447.007.png)



更改256

![](Pic/Aspose.Words.6faf1351-b469-40d1-813b-aca201e8a447.008.png)

![](Pic/Aspose.Words.6faf1351-b469-40d1-813b-aca201e8a447.009.png)



#### Conclusion:

更改model的block數之後根據loss vs iterations發現train loss是隨block數字變大而上升，而在block數等於50時，train loss 跟 validation loss的差距比前面兩個還大，表示有過度擬和的現象發生，在block 1 & 10 兩者之間train loss值大約相同，但block 10的validation loss值比block 1低表示模型具有好的泛化能力，可以更好的擬和未見過的數據。

更改model內block的層數當更改為16時模型的參數數量減少，特徵的表現能力會減弱，可以降低過度擬和的風險，但也會因此可能造成準確率下降，而當更改為256時增加了模型參數，將輸入特徵增加，若是增加太多可能造成過度擬和發生，根據這兩項更改發現更改為256使得train loss下降且validation也有所下降，是更好的結果

#### Discussion:

第二題的更改model的block數一開始更改會報錯，與同組同學討論後得到解決。
