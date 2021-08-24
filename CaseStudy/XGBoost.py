from xgboost import XGBClassifier


xgbmodel = XGBClassifier(learning_rate=0.1,
                      n_estimators=100,         # 树的个数--100棵树建立xgboost
                      max_depth=6,               # 树的深度
                      min_child_weight = 1,      # 叶子节点最小权重
                      gamma=0.,                  # 惩罚项中叶子结点个数前的参数
                      subsample=0.8,             # 随机选择80%样本建立决策树
                      colsample_btree=0.8,       # 随机选择80%特征建立决策树
                      objective='binary:logistic', # 指定损失函数
                      # random_state=27,            # 随机数种子
                      scale_pos_weight=1         # 解决样本个数不平衡的问题 例如，当正负样本比例为1:10时，scale_pos_weight=10
                      )
