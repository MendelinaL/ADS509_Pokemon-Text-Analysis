import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib




app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    mlb_abilities = MultiLabelBinarizer()
    abilities_encoded = mlb_abilities.fit_transform(pokemon_data_encoded['abilities'])
    abilities_encoded_df = pd.DataFrame(abilities_encoded, columns=mlb_abilities.classes_)
    pokemon_data_encoded = pd.concat([pokemon_data_encoded.drop('abilities', axis=1), abilities_encoded_df], axis=1)

    # MultiLabelBinarizer encode 'moves'
    mlb_moves = MultiLabelBinarizer()
    moves_encoded = mlb_moves.fit_transform(pokemon_data_encoded['moves'])
    moves_encoded_df = pd.DataFrame(moves_encoded, columns=mlb_moves.classes_)
    pokemon_data_encoded = pd.concat([pokemon_data_encoded.drop('moves', axis=1), moves_encoded_df], axis=1)
    #Alternative Usage of Saved Model
    #joblib.dump(clf, 'Pokemon_modell.pkl')
    Pokemon_model = open('Pokemon_modell.pkl','rb')
    clf = joblib.load(Pokemon_model)


    clf1 = RandomForestClassifier(random_state=42)
    param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
    }
    grid_search = GridSearchCV(estimator = clf1, param_grid = param_grid, 
                           cv = 3, n_jobs = -1, verbose = 2)
    
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = grid_search.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
    app.run(debug=True)
