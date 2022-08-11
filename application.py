import numpy as np
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))
state_list = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California',
       'Colorado', 'Connecticut', 'Delaware', 'District of Columbia',
       'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana',
       'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
       'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi',
       'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire',
       'New Jersey', 'New Mexico', 'New York', 'North Carolina',
       'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania',
       'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee',
       'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
       'West Virginia', 'Wisconsin', 'Wyoming', 'Puerto Rico']

statefips_list = [ 1,  2,  4,  5,  6,  8,  9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20,
       21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
       38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56,
       72]

@app.route('/')
def homepage():
    return render_template('index.html', state_list=state_list)

@app.route('/predict', methods=['GET','POST'])
def predict():
    state_index =  state_list.index(request.form.get('state'))
    statefips = statefips_list[state_index]
    lat_tract = int(request.form.get('lat_tract'))
    long_tract = int(request.form.get('long_tract'))
    population = int(request.form.get('population'))
    adj_radiuspop_5 = int(request.form.get('adj_radiuspop_5'))
    
    int_features = [statefips, lat_tract, long_tract, population, adj_radiuspop_5]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    if prediction > 7:
        output = 'democrats are likely to win because its more urban'
    else:
        output = 'republicans are likely to win because its more rural'
    return render_template('index.html', state_list=state_list, prediction=f'the prediction is {prediction}', output=output)

if __name__=='__main__':
    app()
