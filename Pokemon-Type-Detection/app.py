import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from flask import Flask,render_template,url_for,request, redirect, render_template
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib



app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/stats', methods = ['POST', 'GET'])
def stats():
    if request.method == 'POST':
        Final_Data = pd.read_csv('Final_Data.csv', index_col = [0])
        pokemon = request.form.get('comment')
        print(pokemon)
        pokemon = str(pokemon)
        pokemon_output = Final_Data.loc[Final_Data['pokemon'] == pokemon]
        return render_template('result.html',tables=[pokemon_output.to_html()],titles = pokemon)
    return 
    

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        Pokemon_model = open('Pokemon_modell.pkl','rb')
        clf = joblib.load(Pokemon_model)
        Final_Data = pd.read_csv('Final_Data.csv', index_col = [0])
        pokemon_data_encoded = Final_Data.copy()

        # Handle missing 'moves' data
        pokemon_data_encoded['moves'] = pokemon_data_encoded['moves'].apply(lambda x: x if isinstance(x, list) else [])

        # MultiLabelBinarizer encode 'abilities'
        mlb_abilities = MultiLabelBinarizer()
        abilities_encoded = mlb_abilities.fit_transform(pokemon_data_encoded['abilities'])
        abilities_encoded_df = pd.DataFrame(abilities_encoded, columns=mlb_abilities.classes_)
        pokemon_data_encoded = pd.concat([pokemon_data_encoded.drop('abilities', axis=1), abilities_encoded_df], axis=1)

        # MultiLabelBinarizer encode 'moves'
        mlb_moves = MultiLabelBinarizer()
        moves_encoded = mlb_moves.fit_transform(pokemon_data_encoded['moves'])
        moves_encoded_df = pd.DataFrame(moves_encoded, columns=mlb_moves.classes_)
        pokemon_data_encoded = pd.concat([pokemon_data_encoded.drop('moves', axis=1), moves_encoded_df], axis=1)

        # Separate features from the target
        X = pokemon_data_encoded.drop('types', axis=1)  
        y = Final_Data['types']  

        # MultiLabelBinarizer encode 'types'
        mlb_types = MultiLabelBinarizer()
        y_encoded = mlb_types.fit_transform(y)

        # Split 
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        # One hot encoding
        X_train = pd.get_dummies(X_train)
        X_test = pd.get_dummies(X_test)

        # Align 
        X_train, X_test = X_train.align(X_test, join='left', axis=1)

        # Handle NaN 
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)

        # RFC
        clf = RandomForestClassifier(random_state=42)

        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }

        # Grid search
        grid_search = GridSearchCV(estimator = clf, param_grid = param_grid, 
                                cv = 3, n_jobs = -1, verbose = 2)

        grid_search.fit(X_train, y_train)

        # Get best estimator
        best_clf = grid_search.best_estimator_

        # Predictions
        y_pred = best_clf.predict(X_test)
        message = request.form.get('comment')
        data = [message]
        vect = grid_search.transform(data).toarray()
        my_prediction = clf.predict(vect)
        return render_template('result.html',prediction = my_prediction)
    return



if __name__ == '__main__':
    app.run(host= '0.0.0.0', port = 5001, debug=True)