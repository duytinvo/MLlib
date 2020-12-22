import os
import time
import sys
import argparse
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve

from sklearn.naive_bayes import MultinomialNB  # NB
from sklearn.neighbors import KNeighborsClassifier  # k-NN
from sklearn.linear_model import SGDClassifier  # logistic regression
from sklearn.tree import DecisionTreeClassifier  # DT
from sklearn.svm import LinearSVC  # linear SVM
from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier


class MLcls:
    def __init__(self, args):
        self.args = args
        self.settings = []
        self.parameters = {}
        self.pipeline = None
        self.best_model = None
        pass

    @staticmethod
    def pkl_write(data, filename='data.pickle'):
        with open(filename, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def pkl_read(filename='data.pickle'):
        with open(filename, 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            data = pickle.load(f)
        return data

    @staticmethod
    def save(model, mfile):
        print("- Save the model...")
        MLcls.pkl_write(model, mfile)
        print("\t+ Done.")

    @staticmethod
    def load(mfile):
        print("- Load the model...")
        model = MLcls.pkl_read(mfile)
        print("\t+ Done.")
        return model

    def vectorization(self, vec_type="tfidf"):
        vec_para = {
                    'vectorizer__strip_accents': ['ascii', 'unicode', None],
                    # 'vectorizer__lowercase': (True, False),
                    'vectorizer__ngram_range': list(zip([1] * 3, list(range(1, 4)))),
                    # 'vectorizer__ngram_range': [(1, 3)],  # onlu use tri-gram
                    'vectorizer__analyzer': ['char', 'word', 'char_wb'],
                    'vectorizer__min_df': [0.0001, 0.001, 0.01, 0.1],
                    'vectorizer__max_df': [0.9999, 0.999, 0.99, 0.9],
                    'vectorizer__binary': (True, False)
                    }
        self.parameters.update(vec_para)

        if vec_type == "hash":
            self.settings += [('vectorizer', HashingVectorizer())]
            vec_para = {'vectorizer__norm': ('l1', 'l2'),
                        'vectorizer__alternate_sign': (True, False)
                        }

        elif vec_type == "tfidf":
            self.settings += [('vectorizer', TfidfVectorizer())]
            vec_para = {
                        'vectorizer__norm': ('l1', 'l2'),
                        'vectorizer__use_idf': (True, False),
                        'vectorizer__smooth_idf': (True, False),
                        'vectorizer__sublinear_tf': (True, False)
                        }
        else:
            # DEFAULT: BOW counting
            self.settings += [('vectorizer', CountVectorizer())]
        self.parameters.update(vec_para)

    def feature_nomalization(self):
        print("- Build the feature nomalization...")
        self.settings += [('scaler', StandardScaler())]
        norm_para = {'scaler__with_mean': (True, False),
                     'scaler__with_std': (True, False),
                     }
        self.parameters.update(norm_para)

    def classifier(self, ml_type="NB"):
        if ml_type == "kNN":
            classifier = KNeighborsClassifier(n_neighbors=5)
            cls_para = {"classifier__n_neighbors": [5, 6, 7, 8, 9, 10],
                        "classifier__weights": ['uniform', 'distance']
                        }
        elif ml_type == "LR":
            # Logistic Regression
            classifier = SGDClassifier(verbose=5, max_iter=1000)
            cls_para = {"classifier__loss": ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
                        "classifier__penalty": ['l1', 'l2', 'elasticnet'],
                        "classifier__alpha": [0.0001, 0.005, 0.001],
                        "classifier__class_weight": ["balanced", None],
                        "classifier__warm_start": [True, False],
                        "classifier__average": [True, False],
                        }
        elif ml_type == "DT":
            classifier = DecisionTreeClassifier(criterion="entropy", random_state=0)
            cls_para = {"classifier__criterion": ['gini', 'entropy'],
                        "classifier__class_weight": ["balanced", None]
                        }
        elif ml_type == "SVM":
            classifier = LinearSVC(verbose=5, max_iter=1000)
            cls_para = {"classifier__penalty": ['l1', 'l2'],
                        "classifier__loss": ['hinge', 'squared_hinge'],
                        "classifier__dual": [True, False],
                        "classifier__C": [0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
                        "classifier__class_weight": ["balanced", None],
                        }
        elif ml_type == "MLP":
            classifier = MLPClassifier(random_state=1, max_iter=100)
            cls_para = {}
        elif ml_type == "AB":
            classifier = AdaBoostClassifier()
            cls_para = {"classifier__n_estimators": [10, 50, 100]}
        elif ml_type == "GB":
            classifier = GradientBoostingClassifier(verbose=5)
            cls_para = {"classifier__n_estimators": [10, 50, 100],
                        "classifier__max_depth": [3, 5, 10, 15],
                        "classifier__warm_start": [True, False]
                        }
        elif ml_type == "RF":
            classifier = RandomForestClassifier(n_estimators=100, verbose=5)
            cls_para = {"classifier__n_estimators": [10, 50, 100],
                        "classifier__criterion": ['gini', 'entropy'],
                        "classifier__class_weight": ["balanced", None]
                        }
        else:
            # DEFAULT: NB
            classifier = MultinomialNB()
            cls_para = {"classifier__alpha": [0.01, 0.1, 1.0, 5.0]}

        self.settings += [('classifier', classifier)]
        self.parameters.update(cls_para)

    def build(self):
        start = time.time()
        print("\t(1) Build the Vectorization...")
        self.vectorization(vec_type=self.args.vec_type)

        # scaler cannot use with NB (MultinomialNB)
        if self.args.scaler and self.args.ml_type != "NB":
            print("\t(2) Build the feature nomalization...")
            self.feature_nomalization()

        print("\t(3) Construct the classifier ...")
        self.classifier(ml_type=self.args.ml_type)
        self.pipeline = Pipeline(self.settings)
        end = time.time()
        print("\t+ Done: %.4f(s)" % (end - start))

    @staticmethod
    def read_data(train_file, dev_file):
        data_train = pd.read_csv(train_file, delimiter="\t").sample(frac=1).reset_index(drop=True)
        data_dev = pd.read_csv(dev_file, delimiter="\t").sample(frac=1).reset_index(drop=True)
        data_merge = pd.concat([data_train, data_dev])
        dev_fold = [-1] * len(data_train) + [0] * len(data_dev)
        x_traindev, y_traindev = data_merge["text"].to_numpy(), data_merge["label"].to_numpy()
        return x_traindev, y_traindev, dev_fold

    def train(self):
        print("- Design the baseline")
        self.build()
        print("- Read train and dev data sets")
        x_traindev, y_traindev, dev_fold = MLcls.read_data(train_file=self.args.train_file,
                                                           dev_file=self.args.dev_file)

        print("- Train the baseline...")
        start = time.time()
        model = GridSearchCV(self.pipeline, self.parameters, cv=PredefinedSplit(test_fold=dev_fold),
                             verbose=5, scoring='f1_weighted')
        model.fit(x_traindev, y_traindev)
        end = time.time()
        print("\t+ Done: %.4f(s)" % (end - start))
        self.best_model = model.best_estimator_
        MLcls.save(self.best_model, self.args.model_name)

    @staticmethod
    def class_metrics(y_true, y_pred):
        acc = metrics.accuracy_score(y_true, y_pred)
        f1_ma = metrics.precision_recall_fscore_support(y_true, y_pred, average='macro')
        f1_we = metrics.precision_recall_fscore_support(y_true, y_pred, average='weighted')
        f1_no = metrics.precision_recall_fscore_support(y_true, y_pred, average=None)
        print("\t+ Accuracy: %.4f(%%)" % (acc * 100))
        measures = {"acc": acc, "prf_macro": f1_ma, "prf_weighted": f1_we, "prf_individual": f1_no}
        return measures

    @staticmethod
    def evaluate(test_file, model_name):
        data_test = pd.read_csv(test_file, delimiter="\t").sample(frac=1).reset_index(drop=True)
        x_test, y_true = data_test["text"].to_numpy(), data_test["label"].to_numpy()
        model = MLcls.load(model_name)
        print("- Evaluate the baseline...")
        start = time.time()
        y_pred = model.predict(x_test)
        mtrcs = MLcls.class_metrics(y_true, y_pred)
        plot_confusion_matrix(model, x_test, y_true)
        plot_roc_curve(model, x_test, y_true)
        end = time.time()
        print("\t+ Done: %.4f(s)" % (end - start))
        return mtrcs

    @staticmethod
    def predict(sent, model_name):
        model = MLcls.load(model_name)
        label = model.predict([sent]).tolist()[0]
        prob = model.predict_proba([sent]).max()
        print("- Inference...")
        print("\t+ %s with p=%.4f" % (label, prob))
        return label, prob


if __name__ == '__main__':
    """
    python baselines.py --train_file /media/data/langID/small_scale/train.csv --dev_file /media/data/langID/small_scale/dev.csv --test_file /media/data/langID/small_scale/test.csv --model_name ./results/small.NB.m --ml_cls NB
    """
    argparser = argparse.ArgumentParser(sys.argv[0])
    
    argparser.add_argument('--train_file', help='Trained file', type=str,
                           default="../../data/vinsenti/dataset/train_full.csv")
    
    argparser.add_argument('--dev_file', help='Developed file', type=str,
                           default="../../data/vinsenti/dataset/dev_full.csv")
    
    argparser.add_argument('--test_file', help='Tested file', type=str,
                           default="../../data/vinsenti/dataset/dev_full.csv")

    argparser.add_argument('--model_dir', help='Model dir', type=str,
                           default="../../data/vinsenti/trained_model/")

    argparser.add_argument("--vec_type", default="tfidf", choices=["count", "tfidf", "hash"], help="vectorization methods")
    
    argparser.add_argument("--scaler", action='store_true', default=False, help="scale flag")
    
    argparser.add_argument('--ml_type', help='Machine learning algorithms', default="SVM", type=str)

    args = argparser.parse_args()
    
    model_dir, _ = os.path.split(args.model_dir)

    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    args.model_name = os.path.join(args.model_dir, args.ml_type + ".pickle")

    trad_ml = MLcls(args=args)
    trad_ml.train()

    # measures = test(args, args.model_name)
    # label, prob = predict("call us to win a price", args.model_name)


