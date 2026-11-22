import pandas as pd, numpy as np, joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay

df = pd.read_csv("loan_data.csv")
# Limpeza: idades irreais e experiência maior que a idade possível
n0=len(df)
df = df[(df.person_age<=100) & (df.person_emp_exp <= df.person_age-14)].reset_index(drop=True)
print("removidos:", n0-len(df))

X, y = df.drop(columns="loan_status"), df["loan_status"]
cat = X.select_dtypes(exclude="number").columns.tolist()
num = X.select_dtypes(include="number").columns.tolist()
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)

pre = ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),cat)],remainder="passthrough")
pipe = Pipeline([("pre",pre),("rf",RandomForestClassifier(random_state=42,n_jobs=-1,class_weight="balanced"))])

cv = StratifiedKFold(5,shuffle=True,random_state=42)
pass

grid = {"rf__n_estimators":[100,150],"rf__max_depth":[10,15,20],
        "rf__min_samples_leaf":[1,2,4],"rf__max_features":["sqrt","log2",0.5]}
search = RandomizedSearchCV(pipe,grid,n_iter=5,scoring="f1",cv=StratifiedKFold(3,shuffle=True,random_state=42),random_state=42,n_jobs=1)
search.fit(Xtr,ytr)
print("melhores params:",search.best_params_, "CV F1:",round(search.best_score_,4))

best = search.best_estimator_
pred = best.predict(Xte); proba = best.predict_proba(Xte)[:,1]
print(classification_report(yte,pred,target_names=["Rejeitado","Aprovado"]))
print("ROC AUC:",round(roc_auc_score(yte,proba),4))

fig,ax=plt.subplots(1,3,figsize=(17,5))
ConfusionMatrixDisplay.from_predictions(yte,pred,ax=ax[0],colorbar=False); ax[0].set_title("Matriz de confusão")
RocCurveDisplay.from_predictions(yte,proba,ax=ax[1]); ax[1].set_title("Curva ROC")
names = best.named_steps["pre"].get_feature_names_out()
imp = pd.Series(best.named_steps["rf"].feature_importances_,index=[n.split("__")[-1] for n in names]).sort_values().tail(12)
imp.plot.barh(ax=ax[2]); ax[2].set_title("Top 12 importâncias")
plt.tight_layout(); plt.savefig("avaliacao.png",dpi=110)
print(imp.sort_values(ascending=False).head(6))
joblib.dump(best,"modelo_random_forest.joblib")
