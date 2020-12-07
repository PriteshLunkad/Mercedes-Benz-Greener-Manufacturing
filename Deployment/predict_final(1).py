from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from docx import Document 
import pandas as pd 
import pickle
import numpy as np
from datetime import datetime
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings('ignore')


app = Flask(__name__,static_url_path = '', static_folder = 'static')
fe = open('static/errors.txt','w+')
error = 0

def read_file(filename):
    global error
    data = {} 
    f = open(filename,'r')
    lines = f.readlines()

    for i in range(0,len(lines)):
        val = lines[i]
        val = val.strip().split(",")
        if (i>=1 and i<=8):
            data[val[0]] = val[1]
        else:
            try:
                data[val[0]] = int(val[1])
            except:
                print("Error Occured"+val[0])
                fe.write(val[0]+" must be a integer"+"\n")
                error = 1
                
    for i in data:
        if (i == 'ID'):
            if(data[i]>10000):
                fe.write(i+" Should be less than 10000 \n")
                error = 1
        elif (i in ['X0','X1','X2','X3','X4','X5','X6','X8']):
            if (len(data[i])>=3):
                fe.write(i+" Should be a categorical variable whose length should be less than 3 \n")
                error = 1
        else:
            if (data[i] not in [0,1]):
                fe.write(i+" Should be a binary variable with either 0 or 1 \n")
                error = 1
    f.close()
    return pd.Series(data)

def final_fun_1(X):
    # Saving the starting time
    start = datetime.now()
    y = [0]
    global error
    # Converting the pandas series into dataframe if a single value is passed
    if len(X.shape) == 1:
        X = X.to_frame().T
    
    # Label encoding categorical features
    try:
        X['X0'] = (pickle.load(open('X0-LabelEncoder.pkl','rb'))).transform(X['X0'])
    except:
        fe.write("Input X0 value is not present in test data. Please enter a valid X0 value\n")
        error = 1

    try:
        X['X1'] = (pickle.load(open('X1-LabelEncoder.pkl','rb'))).transform(X['X1'])
    except:
        fe.write("Input X1 value is not present in test data. Please enter a valid X1 value\n")
        error = 1

    try:
        X['X2'] = (pickle.load(open('X2-LabelEncoder.pkl','rb'))).transform(X['X2'])
    except:
        fe.write("Input X2 value is not present in test data. Please enter a valid X2 value\n")
        error = 1

    try:
        X['X3'] = (pickle.load(open('X3-LabelEncoder.pkl','rb'))).transform(X['X3'])
    except:
        fe.write("Input X3 value is not present in test data. Please enter a valid X3 value\n")
        error = 1

    try:
        X['X4'] = (pickle.load(open('X4-LabelEncoder.pkl','rb'))).transform(X['X4'])
    except:
        fe.write("Input X4 value is not present in test data. Please enter a valid X4 value\n")
        error = 1

    try:
        X['X5'] = (pickle.load(open('X5-LabelEncoder.pkl','rb'))).transform(X['X5'])
    except:
        fe.write("Input X5 value is not present in test data. Please enter a valid X5 value\n")
        error = 1

    try:
        X['X6'] = (pickle.load(open('X6-LabelEncoder.pkl','rb'))).transform(X['X6'])
    except:
        fe.write("Input X6 value is not present in test data. Please enter a valid X6 value\n")
        error = 1

    try:
        X['X8'] = (pickle.load(open('X8-LabelEncoder.pkl','rb'))).transform(X['X8'])
    except:
        fe.write("Input X8 value is not present in test data. Please enter a valid X8 value\n")
        error = 1
    
    try:
        # Creating Summation features
        X29_X127 = X['X29'] + X['X127']
        X54_X127 = X['X54'] + X['X127']
        X118_X314 = X['X118'] + X['X314']
        X119_X314 = X['X119'] + X['X314']
        X127_X162 = X['X127'] + X['X162']
        X127_X166 = X['X127'] + X['X166']
        X127_X272 = X['X127'] + X['X272']
        X127_X276 = X['X127'] + X['X276']
        X127_X328 = X['X127'] + X['X328']
        X136_X261 = X['X136'] + X['X261']
        X136_X314 = X['X136'] + X['X314']
        X221_X314 = X['X221'] + X['X314']
        X261_X263 = X['X261'] + X['X263']
        X261_X315 = X['X261'] + X['X315']
        X263_X314 = X['X263'] + X['X314']
        X314_X315 = X['X314'] + X['X315']
    except:
        error = 1

    try:
        # Dropping columns that are not required
        X = X.drop(pickle.load(open('total_columns_dropped.pkl','rb')),axis = 1)
    except:
        error = 1

    try:
        
        # Adding dimensionality reduction features
        pca = (pickle.load(open('pca.pkl','rb'))).transform(X)
        tsvd= (pickle.load(open('tsvd.pkl','rb'))).transform(X)
        ica = pd.DataFrame((pickle.load(open('ica.pkl','rb'))).transform(X))
        grp = pd.DataFrame((pickle.load(open('grp.pkl','rb'))).transform(X))
        srp = pd.DataFrame((pickle.load(open('srp.pkl','rb'))).transform(X))
        nmf = pd.DataFrame((pickle.load(open('nmf.pkl','rb'))).transform(X))
        fag = pd.DataFrame((pickle.load(open('fag.pkl','rb'))).transform(X))

        X['pca_0'] = pca
        X['tsvd_0'] = tsvd
        for i in range(12):
            X['ica_'+str(i+1)] = ica.loc[:,i]
            X['grp_'+str(i+1)] = grp.loc[:,i]
            X['srp_'+str(i+1)] = srp.loc[:,i]
            X['nmf_'+str(i+1)] = nmf.loc[:,i]
            X['fag_'+str(i+1)] = fag.loc[:,i]
    except:
        error = 1

    try:
        
        # Adding the summation features
        X['X29_plus_X127'] = X29_X127 
        X['X54_plus_X127'] = X54_X127
        X['X118_plus_X314'] = X118_X314
        X['X119_plus_X314'] = X119_X314
        X['X127_plus_X162'] = X127_X162
        X['X127_plus_X166'] = X127_X166
        X['X127_plus_X272'] = X127_X272
        X['X127_plus_X276'] = X127_X276
        X['X127_plus_X328'] = X127_X328
        X['X136_plus_X261'] = X136_X261
        X['X136_plus_X314'] = X136_X314
        X['X221_plus_X314'] = X221_X314
        X['X261_plus_X263'] = X261_X263
        X['X261_plus_X315'] = X261_X315
        X['X263_plus_X314'] = X263_X314
        X['X314_plus_X315'] = X314_X315
    except:
        error = 1
    
    try:
        # Predicting the results using LGBM model
        y = np.exp((pickle.load(open('lgbm_model.pkl','rb'))).predict(X.values))
    except:
        error = 1
    
    # Time taken to predict the value
    end = datetime.now()
    print('Time taken to predict the value: ',end - start) 
    
    # Returning the predicted value
    return y

@app.route('/')
def upload():
   return render_template('upload.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        data = read_file(secure_filename(f.filename))
        y = final_fun_1(data)
        fe.close()
        if(error == 0):
            return 'The Predicted time taken on test bench for car of configuration as given in word document is '+str(y[0])+' secs'
        else:
            return render_template('error.html')

		
if __name__ == '__main__':
   app.run(debug = True) 