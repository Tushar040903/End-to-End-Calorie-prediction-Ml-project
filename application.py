from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    # Render the welcome page (index.html)
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        # Render the input form page (home.html)
        return render_template('home.html')
    else:
        # Create a CustomData instance from form values
        data = CustomData(
            User_ID = int(request.form.get('User_ID')),
            Gender = request.form.get('Gender'),
            Age = float(request.form.get('Age')),
            Height = float(request.form.get('Height')),
            Weight = float(request.form.get('Weight')),
            Duration = float(request.form.get('Duration')),
            Heart_Rate = float(request.form.get('Heart_Rate')),
            Body_Temp = float(request.form.get('Body_Temp'))
        )
        # Convert input to DataFrame
        pred_df = data.get_data_as_data_frame()
        # Make prediction using the pipeline
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        # Render the result page with the predicted calories burnt
        return render_template('result.html', results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0")
