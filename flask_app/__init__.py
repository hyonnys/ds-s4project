from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import os

app = Flask(__name__)
app_root = os.path.dirname(os.path.abspath(__file__))
model_path=os.path.join(app_root, './model/model.pkl')

model=pickle.load(open(model_path, 'rb'))

# from .views import main_views
# app.register_blueprint(main_views.bp)

@app.route("/")                        
def index():
     return render_template('index.html')

@app.route('/submit', methods=['GET', 'POST'])  
def make_prediction():
  features = [int(x) for x in request.form.values()]
  final_features = [np.array(features)]       
  prediction = model.predict(final_features)  
  prediction = prediction[0]    
  return render_template('prediction.html', prediction = prediction) 

# @app.route('/model', methods=['POST'])
# def predict():
#     data=request.get_json(force=True)
#     prediction=model.predict([[np.array(data['exp'])]])
#     output = prediction[0]
#     return jsonify(output)

if __name__ == "__main__":
  app.run(debug=True)