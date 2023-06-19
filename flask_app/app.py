from flask import Flask, request, jsonify, render_template
# from flask_restful import reqparse
import numpy as np
import pickle
import os
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

app = Flask(__name__)

# loading model
app_root = os.path.dirname(os.path.abspath(__file__))
model_path=os.path.join(app_root, './model/model.pkl')

model=pickle.load(open(model_path, 'rb'))

# from .views import main_views
# app.register_blueprint(main_views.bp)

@app.route("/", methods=['GET', 'POST'])                        
def index():
  if request.method == 'GET':
    return render_template('index.html')
  if request.method == 'POST':
    le = LabelEncoder()
    df = pd.DataFrame(columns=['age_cat', 'underlying_disease',\
                 'blood_group', 'gestation',\
                  'cPRA_cat'])
    for x in df:
      df[x] = le.transform(request.form[x])
    
    df['dialysis_duration'] = float(request.form['dialysis_duration'])
    
    pred = model.predict(df)
    return render_template('result.html', prediction=pred)  

#     age_cat = le.transform(request.form['age_cat'])
#     underlying_disease = le.transform(request.form['underlying_disease'])
#     blood_group = le.transform(request.form['blood_group'])
#     gestation = le.transform(request.form['gestation'])
#     cPRA_cat = le.transform(request.form['cPRA_cat'])
#     dialysis_duration = float(request.form['dialysis_duration'])
  
#   waiting_time = 0
#   x = ((age_cat, underlying_disease, blood_group, gestation, cPRA_cat, dialysis_duration), )

# @app.route('/predict', methods=['POST'])
# def predict():
#   for x in request.form.values:
#     if x.select_dtypes(include=['object']).columns
#   le = LabelEncoder()
#   # data = request.get_json
#   # age_cat = data['age_cat']
#   # for x in request.form.values:

#   le = LabelEncoder()
#   age_cat = le.transform(request.form['age_cat'])
#   underlying_disease = le.transform(request.form['underlying_disease'])
#   blood_group = le.transform(request.form['blood_group'])
#   gestation = le.transform(request.form['gestation'])
#   cPRA_cat = le.transform(request.form['cPRA_cat'])
#   dialysis_duration = float(request.form['dialysis_duration'])

#     x = ((age_cat, underlying_disease, blood_group, gestation, cPRA_cat, dialysis_duration), )
#     arr = np.array(x, dtype=np.float32)

#     x_data = arr[0:4]
#     pred=model.predict(x_data)
#     return render_template('result.html', prediction=pred)


# @app.route('/predict', methods=['POST'])  
# def predict():
#   d1 = request.form['age_cat']
#   d2 = request.form['cPRA_cat']
#   d3 = request.form['underlying_disease']
#   d4 = request.form['blood_group']
#   d5 = request.form['gestation']
#   d6 = request.form['dialysis_duration']

#   x = pd.DataFrame([[d1, d2, d3, d4, d5, d6]])
#   le = LabelEncoder()
#   for i in range(5):
#      x[i] = le.transform(x[i])

#   pred=model.predict(x)

#   # features = [int(x) for x in request.form.values()]
#   # final_features = [np.array(features)]       
#   # prediction = model.predict(final_features)      
#   return render_template('result.html', prediction = pred) 

# @app.route('/model', methods=['POST'])
# def predict():
#     data=request.get_json(force=True)
#     prediction=model.predict([[np.array(data['exp'])]])
#     output = prediction[0]
#     return jsonify(output)

if __name__ == "__main__":
  app.run(debug=True)