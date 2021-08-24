# LPI-HyADBS
LPI-HyADBS: A hybrid framework integrating feature extraction based on AdaBoost, and classification models including DNN, XGBoost, and SVM used to predict LPIs
# Data
Data is available at [NONCODE](http://www.noncode.org/), [NPInter v3.0](http://bigdata.ibp.ac.cn/npinter3/index.htm), and [PlncRNADB](http://bis.zju.edu.cn/PlncRNADB/).
# Feature Acquisition
[PyFeat](https://github.com/mrzResearchArena/PyFeat)
# Environment
## python == 3.8.5
## pytorch == 1.4.0
## scikit-learn == 0.23.2
## xgboost == 1.3.1
# Usage
To run the model, default 5 fold cross validation
  python example/main.py
