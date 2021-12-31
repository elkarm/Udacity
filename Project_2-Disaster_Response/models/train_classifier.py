# import libraries
import re
import sys
import os
import subprocess
from sqlalchemy import create_engine
from IPython.display import display
import pandas as pd
import nltk
nltk.download(['punkt','words','stopwords','averaged_perceptron_tagger','maxent_ne_chunker','wordnet'])
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk import pos_tag,ne_chunk
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.base import BaseEstimator, TransformerMixin
import pickle

# inpstall termcolor
subprocess.check_call([sys.executable, "-m", "pip", "install", 'termcolor'])
print("termcolor installed")
from termcolor import colored, cprint


def load_data(database_filepath):
    """Load clean dataset from database
    
    inputs:
    filepath: string. Filepath for database file containing messages dataset.
       
    outputs:
    df: dataframe. Dataframe containing clean dataset of messages.
    """

    engine = create_engine('sqlite:///' + database_filepath)
    table_name = database_filepath.split('/')[-1]
    table_name = table_name.split('.')[0]
    df = pd.read_sql_table(table_name,engine)
    
    X = df['message']
    y = df.iloc[:,4:]
    
    
    category_names = y.columns 
    return X, y, category_names
    
   # display(df.head())
    #return df
    
    

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    
    

def tokenize(text):
    
    # in the message text identify the url to replace it. Expression
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # 
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    


def build_model():
    """ build pipeline with transformers ans classifier for the model
    
    INPUT: None
    OUTPUT: model pipeline with tunes parameters"""
    
    # create pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
   ])
    

    # possible parameters for the pipine to be tuned
    parameters = {
        'tfidf__use_idf': (True, False),
        'tfidf__smooth_idf': [True, False],
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 4]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    
    """function to evaluate the precision, recall and fscore of the multiple X variables
    
       INPUT: model object, X-test dataset, y-test dataset and column names
       OUTPUT: prints the presicion of the y predicted variables"""
    
    
    # calculate the predicted y - y_pred
    y_pred = model.predict(X_test)
    
    # get -false-positives of predicted y compared to test y and get accuracy

    
    for i, col in enumerate(category_names):
        precision, recall, fscore, support = precision_recall_fscore_support(y_test[col],
                                                                    y_pred[:, i],
                                                                    average='weighted')

        print('\nReport for ({}) column:'.format(colored(col, 'red', attrs=['bold'])))

        metrics_params = [
            {'metric': precision, 'name': 'Precision'},
            {'metric': recall, 'name': 'Recall'},
            {'metric': fscore, 'name': 'F-score'}
        ]
        for pmetric in metrics_params:
            if pmetric['metric'] >= 0.75:
                print('  - {}: {}'.format(pmetric['name'],colored(round(pmetric['metric'], 2), 'green')))
            else:
                print('  - {}: {}'.format(pmetric['name'],colored(round(pmetric['metric'], 2), 'yellow')))



def save_model(model, model_filepath):
    """
    Save model to a pickle file
    """
    pickle.dump(model, open(model_filepath, 'wb'))
    print("model saved in pickle file")


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!!!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
