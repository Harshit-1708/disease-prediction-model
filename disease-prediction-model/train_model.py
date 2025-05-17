import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

file_id_1 = "1-ewXOznJwFefiMopi41llTnftgmSkTm_"
url_1 = f"https://drive.google.com/uc?export=download&id={file_id_1}"
df = pd.read_csv(url_1)

df.fillna("None", inplace=True)
symptom_cols = [col for col in df.columns if 'Symptom' in col]

for col in symptom_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

le_disease = LabelEncoder()
df['Disease'] = le_disease.fit_transform(df['Disease'])

X = df[symptom_cols]
y = df['Disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

with open('model.pkl', 'wb') as f:
    pickle.dump((model, le_disease), f)

print("Training completed and model saved as model.pkl")