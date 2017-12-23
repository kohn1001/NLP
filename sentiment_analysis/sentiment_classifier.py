from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
import os
import time


def build_classifier(par_path):
    print (par_path)

    classes = ['neg', 'pos']
    training_set = []
    training_lables = []
    test_set = []
    test_lables = []
    # par_path = os.getcwd()


    for curr_class in classes:
        curr_dir = os.path.join(par_path, curr_class)
        print ('curr dir is: ', curr_dir)
        for file_name in os.listdir(curr_dir):

           # if os.di
            with open(os.path.join(curr_dir, file_name)) as f:
                content = f.read()
                if file_name.startswith('cv9'):
                    test_set.append(content)
                    test_lables.append(curr_class)
                else:
                    training_set.append(content)
                    training_lables.append(curr_class)
    vectorizer = TfidfVectorizer(min_df= 5,
                                 max_df= 0.8,
                                 sublinear_tf=True,
                                 use_idf=True)
    print('number of trainig examples: ', len(training_set))
    training_vector = vectorizer.fit_transform(training_set)
    test_vector = vectorizer.transform(test_set)
    classifier_rbf = svm.SVC()
    t0 = time.time()
    classifier_rbf.fit(training_vector, training_lables)
    t1 = time.time()
    prediction_rbf = classifier_rbf.predict(test_vector)
    t2 = time.time()
    time_rbf_train = t1 - t0
    time_rbf_predict = t2-t1
    print("Results for SVC(kernel=rbf)")
    print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
    print(classification_report(test_lables, prediction_rbf))

print ('home dir is at: ', os.getenv("HOME"))
build_classifier(os.path.join(os.getenv("HOME"), 'Downloads', 'review_polarity', 'txt_sentoken'))



