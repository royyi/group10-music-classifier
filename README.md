# Group 10 - Music Classifier
## Repo Summary


We have 4 folders: 
- `dependencies` contains `utils.py`, which comes from https://github.com/mdeff/fma
- `model` contains all codes related to eda and models
- `web_app` contains all codes needed for lauching the web app
- `more_files` contains some unused files

## Lauch web app with local flask backend
### Set up flask backend
#### Do the following steps if you don't have flask installed
1. `cd web_app` to enter `web_app` folder
2. Optional: Create py venv
3. Run `pip install flask` if you haven't installed flask
4. Run `flask run` and check if endpoints are up in http://localhost:5000/

### Lauch app
1. Run `cd frontend` to enter frontend code folder
2. Run `npm install`
3. Run `npm start`
4. See the web interface at http://localhost:3000/

App is ready to use. You can upload audio file in `.wav` format and click `Classify` to see the results.

#### Note
It can take a while or a short
time to classify songs. If it takes longer
that a minute to classify, restart the python and
react programs and try again.
