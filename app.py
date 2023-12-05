from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


application=Flask(__name__)

app=application

## Route for a home page

@app.route('/help')

def index():
    return render_template('help.html')


@app.route('/home',methods=['GET','POST'])
def predict_datapoint():
    if request.method=="GET":
        return render_template('home.html')
    
    else:
        data=CustomData(
            fti=(request.form.get('fti')),
            t3=(request.form.get('t3')),
            tsh=(request.form.get('tsh')),
            tt4=(request.form.get('tt4')),
        )
        pred_df=data.get_data_as_data_frame()

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results)
    


if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)