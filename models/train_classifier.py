{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loading data...\n",
      "    DATABASE: -f\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package wordnet to\n",
      "[nltk_data]     C:\\Users\\wiemb\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package wordnet is already up-to-date!\n",
      "[nltk_data] Downloading package punkt to\n",
      "[nltk_data]     C:\\Users\\wiemb\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package punkt is already up-to-date!\n",
      "[nltk_data] Downloading package stopwords to\n",
      "[nltk_data]     C:\\Users\\wiemb\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package stopwords is already up-to-date!\n"
     ]
    },
    {
     "ename": "ValueError",
     "evalue": "Table disaster_messages_tbl not found",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mInvalidRequestError\u001b[0m                       Traceback (most recent call last)",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\pandas\\io\\sql.py\u001b[0m in \u001b[0;36mread_sql_table\u001b[1;34m(table_name, con, schema, index_col, coerce_float, parse_dates, columns, chunksize)\u001b[0m\n\u001b[0;32m    241\u001b[0m     \u001b[1;32mtry\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 242\u001b[1;33m         \u001b[0mmeta\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mreflect\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0monly\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mtable_name\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mviews\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mTrue\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    243\u001b[0m     \u001b[1;32mexcept\u001b[0m \u001b[0msqlalchemy\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mexc\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mInvalidRequestError\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\sqlalchemy\\sql\\schema.py\u001b[0m in \u001b[0;36mreflect\u001b[1;34m(self, bind, schema, views, only, extend_existing, autoload_replace, resolve_fks, **dialect_kwargs)\u001b[0m\n\u001b[0;32m   4265\u001b[0m                         \u001b[1;34m\"Could not reflect: requested table(s) not available \"\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 4266\u001b[1;33m                         \u001b[1;34m\"in %r%s: (%s)\"\u001b[0m \u001b[1;33m%\u001b[0m \u001b[1;33m(\u001b[0m\u001b[0mbind\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mengine\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0ms\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m\", \"\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mjoin\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mmissing\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   4267\u001b[0m                     )\n",
      "\u001b[1;31mInvalidRequestError\u001b[0m: Could not reflect: requested table(s) not available in Engine(sqlite:///-f): (disaster_messages_tbl)",
      "\nDuring handling of the above exception, another exception occurred:\n",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-6-27df422646d2>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m    143\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    144\u001b[0m \u001b[1;32mif\u001b[0m \u001b[0m__name__\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;34m'__main__'\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 145\u001b[1;33m     \u001b[0mmain\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;32m<ipython-input-6-27df422646d2>\u001b[0m in \u001b[0;36mmain\u001b[1;34m()\u001b[0m\n\u001b[0;32m    118\u001b[0m         \u001b[0mdatabase_filepath\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mmodel_filepath\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0msys\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0margv\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    119\u001b[0m         \u001b[0mprint\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'Loading data...\\n    DATABASE: {}'\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mformat\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdatabase_filepath\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 120\u001b[1;33m         \u001b[0mX\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mY\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mload_data\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdatabase_filepath\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    121\u001b[0m         \u001b[0mX_train\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mX_test\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mY_train\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mY_test\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mtrain_test_split\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mY\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mtest_size\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;36m0.2\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    122\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m<ipython-input-6-27df422646d2>\u001b[0m in \u001b[0;36mload_data\u001b[1;34m(database_filepath)\u001b[0m\n\u001b[0;32m     33\u001b[0m        \"\"\"\n\u001b[0;32m     34\u001b[0m     \u001b[0mengine\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mcreate_engine\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'sqlite:///{}'\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mformat\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdatabase_filepath\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 35\u001b[1;33m     \u001b[0mdf\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mpd\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mread_sql_table\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'disaster_messages_tbl'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mengine\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     36\u001b[0m     \u001b[0mX\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mdf\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'message'\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     37\u001b[0m     \u001b[0mY\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mdf\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdrop\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'id'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m'message'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m'original'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m'genre'\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0maxis\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\pandas\\io\\sql.py\u001b[0m in \u001b[0;36mread_sql_table\u001b[1;34m(table_name, con, schema, index_col, coerce_float, parse_dates, columns, chunksize)\u001b[0m\n\u001b[0;32m    242\u001b[0m         \u001b[0mmeta\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mreflect\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0monly\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mtable_name\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mviews\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mTrue\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    243\u001b[0m     \u001b[1;32mexcept\u001b[0m \u001b[0msqlalchemy\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mexc\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mInvalidRequestError\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 244\u001b[1;33m         \u001b[1;32mraise\u001b[0m \u001b[0mValueError\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34mf\"Table {table_name} not found\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    245\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    246\u001b[0m     \u001b[0mpandas_sql\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mSQLDatabase\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mcon\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mmeta\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mmeta\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mValueError\u001b[0m: Table disaster_messages_tbl not found"
     ]
    }
   ],
   "source": [
    "import sys\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sqlalchemy import create_engine\n",
    "import nltk\n",
    "from nltk.tokenize import word_tokenize\n",
    "from nltk.stem import WordNetLemmatizer\n",
    "from nltk.corpus import stopwords\n",
    "import re\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer\n",
    "from sklearn.ensemble import AdaBoostClassifier\n",
    "from sklearn.multioutput import MultiOutputClassifier\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import classification_report\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "import pickle\n",
    "\n",
    "nltk.download(['wordnet', 'punkt', 'stopwords'])\n",
    "\n",
    "def load_data(database_filepath):\n",
    "    \"\"\"\n",
    "       Function:\n",
    "       load data from database\n",
    "\n",
    "       Args:\n",
    "       database_filepath: the path of the database\n",
    "\n",
    "       Return:\n",
    "       X (DataFrame) : Message features dataframe\n",
    "       Y (DataFrame) : target dataframe\n",
    "       category (list of str) : target labels list\n",
    "       \"\"\"\n",
    "    engine = create_engine('sqlite:///{}'.format(database_filepath))\n",
    "    df = pd.read_sql_table('disaster_messages_tbl', engine)\n",
    "    X=df['message']\n",
    "    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)\n",
    "    return X, Y\n",
    "\n",
    "\n",
    "def tokenize(text):\n",
    "    \"\"\"\n",
    "    Function: split text into words and return the root form of the words\n",
    "    Args:\n",
    "      text(str): the message\n",
    "    Return:\n",
    "      lemm(list of str): a list of the root form of the message words\n",
    "    \"\"\"\n",
    "\n",
    "   # Normalize\n",
    "    text = re.sub(r\"[^a-zA-Z0-9]\", \" \", text) \n",
    "   # Tokenize \n",
    "    words = word_tokenize(text)\n",
    "   # Remove stop_words\n",
    "    words = [w for w in words if w not in stopwords.words(\"english\")]\n",
    "   # Lemmatize\n",
    "    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]\n",
    "    return lemm\n",
    "\n",
    "\n",
    "def build_model():\n",
    "    \"\"\"\n",
    "     Function: build a model for classifing the disaster messages\n",
    "\n",
    "     Return:\n",
    "       cv(list of str): classification model\n",
    "     \"\"\"\n",
    "\n",
    "    # Create a pipeline\n",
    "    pipeline = Pipeline([\n",
    "        ('vect', CountVectorizer(tokenizer = tokenize)),\n",
    "        ('tfidf', TfidfTransformer()),\n",
    "        ('clf',  MultiOutputClassifier(RandomForestClassifier()))\n",
    "                   ])\n",
    "    # Create Grid search parameters\n",
    "    parameters = {\n",
    "        'tfidf__use_idf': (True, False),\n",
    "        'clf__estimator__n_estimators': [50, 60, 70]\n",
    "    }\n",
    "\n",
    "    cv = GridSearchCV(pipeline, param_grid=parameters)\n",
    "\n",
    "    return cv\n",
    "\n",
    "\n",
    "def evaluate_model(model, X_test, Y_test):\n",
    "    \"\"\"\n",
    "    Function: Evaluate the model and print the f1 score, precision and recall for each output category of the dataset.\n",
    "    Args:\n",
    "    model: the classification model\n",
    "    X_test: test messages\n",
    "    Y_test: test target\n",
    "    \"\"\"\n",
    "    y_pred = model.predict(X_test)\n",
    "    i = 0\n",
    "    for col in Y_test:\n",
    "        print('Feature {}: {}'.format(i + 1, col))\n",
    "        print(classification_report(Y_test[col], y_pred[:, i]))\n",
    "        i = i + 1\n",
    "    accuracy = (y_pred == Y_test.values).mean()\n",
    "    print('The model accuracy is {:.3f}'.format(accuracy))\n",
    "\n",
    "\n",
    "def save_model(model, model_filepath):\n",
    "    \"\"\"\n",
    "    Function: Save a pickle file of the model\n",
    "    Args:\n",
    "    model: the classification model\n",
    "    model_filepath (str): the path of pickle file\n",
    "    \"\"\"\n",
    "\n",
    "    with open(model_filepath, 'wb') as f:\n",
    "        pickle.dump(model, f)\n",
    "\n",
    "\n",
    "def main():\n",
    "    if len(sys.argv) == 3:\n",
    "        database_filepath, model_filepath = sys.argv[1:]\n",
    "        print('Loading data...\\n    DATABASE: {}'.format(database_filepath))\n",
    "        X, Y = load_data(database_filepath)\n",
    "        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)\n",
    "        \n",
    "        print('Building model...')\n",
    "        model = build_model()\n",
    "        \n",
    "        print('Training model...')\n",
    "        model.fit(X_train, Y_train)\n",
    "        \n",
    "        print('Evaluating model...')\n",
    "        evaluate_model(model, X_test, Y_test)\n",
    "\n",
    "        print('Saving model...\\n    MODEL: {}'.format(model_filepath))\n",
    "        save_model(model, model_filepath)\n",
    "\n",
    "        print('Trained model saved!')\n",
    "\n",
    "    else:\n",
    "        print('Please provide the filepath of the disaster messages database '\\\n",
    "              'as the first argument and the filepath of the pickle file to '\\\n",
    "              'save the model to as the second argument. \\n\\nExample: python '\\\n",
    "              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')\n",
    "\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    main()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": "null",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}