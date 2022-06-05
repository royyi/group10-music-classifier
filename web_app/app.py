from get_prediction import *
from flask import Flask, request
from datetime import timedelta
from werkzeug.utils import secure_filename
from flask_cors import CORS

# Configuration stuff
app = Flask(__name__)

# Cannot send request to localhost:5000 from localhost:3000 without this line
CORS(app)

# Set secret key thats needed for some reason
app.config['SECRET_KEY'] = 'bobross'
#app.secret_key = "bobross"

# Only delete webpage data after 5 days
app.permanent_session_lifetime = timedelta(days=5)

#Class to store prediction
class Concurrency:
    prediction = "Upload Audio File"

# Home page of website
@app.route("/home", methods=['GET', "POST"])
def home():
    
    # Function that checks if we uploaded files        
    if request.method == 'POST':

        #Retreive file
        f = request.files['file']

        #Get prediction
        prediction = get_prediction(f)

        #Print prediction and file name
        print("prediction", prediction)
        print("filename: ", f.filename)

        # Trim prediction to only be relevant text
        Concurrency.prediction = prediction[2:-2]
        
        # Return JSON that contains prediction to front end
        return {'text': Concurrency.prediction}
    
    else:

        # Return JSON that contains prediction to front end
        return {'text': Concurrency.prediction}

if __name__ == '__main__':
    # Allows for live debugging and updating
    app.run(debug=True)