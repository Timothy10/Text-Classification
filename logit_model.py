import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


#file locations - these should point to locations of the Excel files being used for input/output. these locations can be modifed as needed.
#import data to be analyzed
import_link = r"C:/Users/tbeauchamp/Desktop/text_data.xlsx"

#output data with model predictions
output_predictions_link = r"C:/Users/tbeauchamp/Desktop/Output.xlsx"

#output model features and weights
output_features = r"C:/Users/tbeauchamp/Desktop/model_features.xlsx"

#output results from parameter tuning
output_tuning_results = r"C:/Users/tbeauchamp/Desktop/Logit_results.xlsx"



def import_data(file_location):
    # imports data and creates the fulltext field - concatenation of title and bodytext separated by a white space
    data = pd.read_excel(io=file_location, sheet_name="Sheet1", header=0, index_col=None, names=['articleID','Status','title','bodytext'])
    data['title'] = data['title'].fillna("")
    data['bodytext'] = data['bodytext'].fillna("")
    data['fulltext'] = data['title'] + ' ' + data['bodytext']
    
    return data

    
def output_data(df_list, file_location):
    #output panda dataframe to an existing excel file, with column names specified to 'Sheet1'
    xlsx_writer = pd.ExcelWriter(file_location)
    counter = 1
    for df in df_list:
        df.to_excel(xlsx_writer, 'Sheet' + str(counter))
        counter += 1
    xlsx_writer.save()


def train_test(data):
    #filter for only relevant and not relevant citations
    Relevant = data[data['Status']=='Accepted']
    Not_Relevant = data[data['Status']=='Rejected']
    R_NR = Relevant.append(Not_Relevant)
    
    # run de-dupe algorithm here
    # Sort by bodytext alphabetically.
    # flag rows that are nearly identical with the either previous row or next row
    # second flag for cases where duplicate sets are inconsistently curated.
    # if all articles in a duplicate set are accepted, then leave one in as accepted
    # if all articles in a duplicate set are rejected, then leave one in as rejected
    # if articles are inconsistently labeled as accepted and rejected remove entire duplicate set from train/test data
    
    #balancing the dataset between relevant and not relevant volume
    R_NR = even_split(R_NR)
    
    #split into training and test sets
    train = R_NR.sample(frac=0.7)
    
    U = R_NR.merge(right=train, how='outer', on='articleID', indicator=True)
    test = U[U['_merge']=='left_only']
    test = test.filter(items=['articleID','Status_x','title_x','bodytext_x','fulltext_x'])
    test = test.rename(index=str, columns = {'Status_x':'Status', 'title_x':'title', 'bodytext_x':'bodytext', 'fulltext_x':'fulltext'})
    
    #return training and test sets
    return train, test


def even_split(data):
    # Input: data to used for train/test
    # Output: subset of input dataset, with the majority label being reduced to balance the count with the minority. This is to avoid biasing he model against the minority label.

    #split of data by status
    Relevant = data[data['Status']=='Accepted']
    Not_Relevant = data[data['Status']=='Rejected']   

    # count number of relevant vs not relevant examples
    agg_count = data.groupby(['Status'])['articleID'].count()
    relevant = agg_count['Accepted']
    rejected = agg_count['Rejected']
    
    frac = relevant/rejected
    if frac > 1:
        majority = 'Accepted'
        minority = 'Rejected'
        sample_frac = agg_count[minority]/agg_count[majority]
        relevant_sample = Relevant.sample(frac=sample_frac)
        combined_data = Not_Relevant.append(relevant_sample)
    elif frac < 1:
        majority = 'Rejected'
        minority = 'Accepted'
        sample_frac = agg_count[minority]/agg_count[majority]
        notrelevant_sample = Not_Relevant.sample(frac=sample_frac)
        combined_data = Relevant.append(notrelevant_sample)
    
    return combined_data
    
    
def tfidf_logit_tune(data):
    # Accepted --> 1
    # Rejected --> 0
    
    vect = TfidfVectorizer(decode_error = 'replace', ngram_range = (1,2), stop_words = 'english')
    logit = linear_model.LogisticRegression(solver='sag')
    pipeline = Pipeline([('tfidf', vect), ('logit', logit)])
    
    max_features_array = np.linspace(start = 1000, stop = 10000, num = 5)
    max_features_array = max_features_array.astype(int)
    max_df_array = np.linspace(start = 0.9, stop = 1.0, num = 3)
    min_df_array = np.linspace(start = 0.0, stop = 0.1, num = 3)
    C_array = np.logspace(start = -1, stop = 2, num = 8)
    #solver_array = ['sag', 'saga', 'liblinear', 'lbfgs']
    
    parameters = {'tfidf__max_df' : max_df_array, 'tfidf__min_df' : min_df_array, 'tfidf__max_features' : max_features_array, 'logit__C' : C_array}
    
    X = data['fulltext']
    
    # replace status labels with binary values
    y = data['Status'].replace('Accepted', 1).replace('Rejected', 0)
    
    classifier = GridSearchCV(pipeline, parameters, return_train_score=True)
    classifier.fit(X, y)
    results = pd.DataFrame(classifier.cv_results_)

    output_data([results], output_tuning_results)
    
    
def tfidf_logit_predict(train, test):
    # Accepted --> 1
    # Rejected --> 0
    
    #tuned features from GridSearchCV
    max_features = 17000
    max_df = 0.9
    min_df = 0.0
    C = 1
    solver = 'sag'
    
    #tf-idf vectorizer
    vect = TfidfVectorizer(decode_error = 'replace', ngram_range = (1,2), stop_words = 'english', max_features = max_features, max_df = max_df, min_df = min_df)
    X = vect.fit_transform(train['fulltext'])
    Xtest = vect.transform(test['fulltext'])
    
    # replace status labels with binary values
    y = train['Status'].replace('Accepted', 1).replace('Rejected', 0)
    ytest = test['Status'].replace('Accepted', 1).replace('Rejected', 0)
    
    logit = linear_model.LogisticRegression(C=C, solver=solver)
    
    accuracy = logit.fit(X, y).score(Xtest, ytest)
    print("model accuracy on test dataset: {:.2%}".format(accuracy))
    
    ytest_scores = logit.predict(Xtest)
    ytest_confidence = logit.predict_proba(Xtest)
    
    #output features with weights from logit model
    output_data([pd.DataFrame(logit.coef_.transpose()), pd.DataFrame(vect.get_feature_names())], output_features)
    
    return accuracy, ytest_scores, ytest_confidence

    
def logit_tuning():
    
    data = import_data(import_link)
    data = even_split(data)
    tfidf_logit_tune(data)    
    
    
def main():
    
    #data set to be imported
    data = import_data(import_link)
    
    train, test = train_test(data)
    
    acc, ytest_scores, ytest_confidence = tfidf_logit_predict(train, test)
    
    test['model scores'] = ytest_scores
    output_data([test, pd.DataFrame(ytest_confidence)], output_predictions_link)


#logit_tuning()
main()