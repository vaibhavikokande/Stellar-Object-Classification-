import pandas as pd 
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import LocalOutlierFactor
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from joblib import dump


data = pd.read_csv("star_classification.csv")


encoder = LabelEncoder()
data['class'] = encoder.fit_transform(data['class'])

#outlier detection

clf = LocalOutlierFactor()
y_pred = clf.fit_predict(data)



x_score = clf.negative_outlier_factor_
outlier_score = pd.DataFrame()
outlier_score["score"] = x_score

#threshold
threshold2 = -1.5                                            
filtre2 = outlier_score["score"] < threshold2
outlier_index = outlier_score[filtre2].index.tolist()

selected_features = ['z','cam_col','i','delta','MJD','plate','spec_obj_ID','alpha','field_ID']
x = data[selected_features]
y = data['class']

sm = SMOTE(random_state=42)
x, y = sm.fit_resample(x, y)
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)




model = XGBClassifier()
model.fit(x, y)


#Save the trained model to a file
dump(model, 'stellar_model.joblib')