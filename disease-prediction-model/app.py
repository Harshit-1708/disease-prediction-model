 from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model, le_disease = pickle.load(f)

file_id_2 = "1-fOjI_XUra9wTrt_wMjVlxlqYkDu7lXR"
file_id_3 = "1-bmwBxQii_BvGXdeUTTutLh9LSUiCpNA"
file_id_4 = "1-fENBbe81B8Ju_5wDckOvRSpwsgoUT2R"

url_2 = f"https://drive.google.com/uc?export=download&id={file_id_2}"
url_3 = f"https://drive.google.com/uc?export=download&id={file_id_3}"
url_4 = f"https://drive.google.com/uc?export=download&id={file_id_4}"

precautions_df = pd.read_csv(url_2)
desc_df = pd.read_csv(url_3)
severity_df = pd.read_csv(url_4)

symptom_cols = [f"Symptom_{i}" for i in range(1, 18)]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        symptoms = []
        for symptom in symptom_cols:
            val = request.form.get(symptom)
            if not val or val.strip() == '':
                val = 'None'
            symptoms.append(val.strip())

        encoded_symptoms = [hash(s) % 10000 for s in symptoms]
        input_data = np.array(encoded_symptoms).reshape(1, -1)

        pred_encoded = model.predict(input_data)[0]
        disease_pred = le_disease.inverse_transform([pred_encoded])[0]

        precs = precautions_df[precautions_df['Disease'].str.lower() == disease_pred.lower()]
        if not precs.empty:
            precautions_list = precs.iloc[0]['Precaution'].split(', ')
        else:
            precautions_list = ["No precautions found"]

        descriptions = []
        severities = []
        for s in symptoms:
            if s.lower() == 'none':
                descriptions.append("No symptom entered")
                severities.append("N/A")
            else:
                desc_row = desc_df[desc_df['Symptom'].str.lower() == s.lower()]
                sev_row = severity_df[severity_df['Symptom'].str.lower() == s.lower()]
                desc = desc_row.iloc[0]['Description'] if not desc_row.empty else "Description not found"
                sev = sev_row.iloc[0]['Severity'] if not sev_row.empty else "Severity not found"
                descriptions.append(desc)
                severities.append(sev)

        symptom_info = list(zip(symptoms, descriptions, severities))

        return render_template('result.html', disease=disease_pred,
                               precautions=precautions_list,
                               symptom_info=symptom_info)

    return render_template('index.html', symptoms=symptom_cols)

if __name__ == '__main__':
    app.run(debug=True)