from xgboost import XGBClassifier


xgbmodel = XGBClassifier(learning_rate=0.1,
                      n_estimators=100,
                      max_depth=6,
                      min_child_weight = 1,
                      gamma=0.,
                      subsample=0.8,
                      colsample_bytree=0.8,
                      objective='binary:logistic',
                      scale_pos_weight=1
                      )
