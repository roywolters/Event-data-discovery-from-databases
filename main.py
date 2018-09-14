from datetime import datetime

from event_data_discovery.activity_identifier_discovery import ActivityIdentifierDiscoverer
from event_data_discovery.activity_identifier_predictors import make_sklearn_pipeline

# ____________________________________________________________________________________________________________________

from sklearn.utils import compute_sample_weight

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.metrics import make_scorer, fbeta_score

from sknn.mlp import Classifier as NNClassifier
from sknn.mlp import Layer

from scipy.stats import randint as sp_randint, expon as sp_expon, uniform as sp_uniform

# ____________________________________________________________________________________________________________________

mimic_files = {
    'openslex': 'data/mimic_demo.slexmm',
    'timestamps': 'mimic/event_timestamps.json',
    'candidates': 'mimic/candidates_in_table.json',
    'feature_values': 'mimic/features_in_table.json',
    # 'candidates': 'mimic/candidates_lookup.json',
    # 'feature_values': 'mimic/features_lookup.json',
    'y_true': 'mimic/ground_truth.json',
}

# ____________________________________________________________________________________________________________________

adw_files = {
    'openslex': 'data/adw-mm.slexmm',
    'timestamps': 'adw/timestamps.json',
    'candidates': 'adw/candidates_in_table.json',
    'feature_values': 'adw/features_in_table.json',
    # 'candidates': 'adw/candidates_lookup.json',
    # 'feature_values': 'adw/features_lookup.json',
    'y_true': 'adw/adw_ground_truth.json',
}

# ____________________________________________________________________________________________________________________

aid_mimic = ActivityIdentifierDiscoverer(mimic_files['openslex'])
# aid_mimic.generate_candidates(mimic_files['timestamps'], ['in_table'])
# aid_mimic.generate_candidates(mimic_files['timestamps'], ['lookup'])
# aid_mimic.save_candidates(mimic_files['candidates'])
aid_mimic.load_candidates(mimic_files['candidates'])
# aid_mimic.compute_features(features='filtered')
# aid_mimic.save_features(mimic_files['feature_values'])
aid_mimic.load_features(mimic_files['feature_values'])
aid_mimic.load_y_true(mimic_files['y_true'])

# ____________________________________________________________________________________________________________________

sample_weights = compute_sample_weight('balanced', aid_mimic.y_true)

tuning_params = [
    {
        'name': "Nearest Neighbors",
        'predictor': make_sklearn_pipeline(KNeighborsClassifier()),
        'parameters': {
            'clf__n_neighbors': sp_randint(2, 20),
            'clf__weights': ['uniform', 'distance'],
        },
        'n_iter': 1000,
        'fit_params': None,
    },
    {
        'name': "Linear SVM",
        'predictor': make_sklearn_pipeline(SVC(kernel="linear", class_weight='balanced', random_state=1)),
        'parameters': {
            'clf__C': sp_expon(),
        },
        'n_iter': 1000,
        'fit_params': None,
    },
    {
        'name': "RBF SVM",
        'predictor': make_sklearn_pipeline(SVC(class_weight='balanced', random_state=1)),
        'parameters': {
            'clf__C': sp_expon(),
        },
        'n_iter': 1000,
        'fit_params': None,
    },
    {
        'name': "Decision Tree",
        'predictor': make_sklearn_pipeline(DecisionTreeClassifier(class_weight='balanced', random_state=1)),
        'parameters': {
            'clf__min_weight_fraction_leaf': sp_uniform(loc=0, scale=0.5),
        },
        'n_iter': 1000,
        'fit_params': None,
    },
    {
        'name': "Random Forest",
        'predictor': make_sklearn_pipeline(RandomForestClassifier(class_weight='balanced', random_state=1)),
        'parameters': {
            'clf__n_estimators': sp_randint(10, 500),
            'clf__max_features': sp_uniform(),
            'clf__min_weight_fraction_leaf': sp_uniform(loc=0, scale=0.5),
        },
        'n_iter': 1000,
        'fit_params': None,
    },
    {
        'name': "AdaBoost",
        'predictor': make_sklearn_pipeline(
            AdaBoostClassifier(DecisionTreeClassifier(class_weight='balanced', random_state=1), random_state=1)),
        'parameters': {
            'clf__n_estimators': sp_randint(10, 500),
            'clf__base_estimator__min_weight_fraction_leaf': sp_uniform(loc=0, scale=0.5),
        },
        'n_iter': 1000,
        'fit_params': None,
    },
    {
        'name': 'Neural Network',
        'predictor': make_sklearn_pipeline(
            NNClassifier(
                layers=[
                    Layer("Rectifier", units=25),
                    Layer("Softmax")
                ],
                n_iter=25,
                random_state=1
            )
        ),
        'parameters': {
            'clf__hidden0__units': sp_randint(2, 50),
            'clf__learning_rate': sp_expon(scale=.001)
        },
        'n_iter': 100,
        'fit_params': {
            'clf__w': sample_weights
        },
    },
]
scoring = {
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'f.5': make_scorer(fbeta_score, beta=2),
    'f2': make_scorer(fbeta_score, beta=2),
}
n_splits = 5
refit = 'f1'
verbose = 1

t0 = datetime.now()
print(str(t0) + ': starting parameter tuning')
aid_mimic.tune_params(tuning_params, scoring, n_splits=n_splits, refit=refit,
                      verbose=verbose)
t1 = datetime.now()
print(str(t1) + ': parameter tuning finished')
print('time elapsed: ' + str(t1 - t0))

# ____________________________________________________________________________________________________________________

aid_mimic.save_tuning_results('tuning_results_all.pkl')
aid_mimic.save_best_predictors('best_predictors_all.pkl')

# ____________________________________________________________________________________________________________________

aid_adw = ActivityIdentifierDiscoverer(adw_files['openslex'])
# aid_adw.generate_candidates(adw_files['timestamps'], ['in_table'])
# aid_adw.generate_candidates(adw_files['timestamps'], ['lookup'])
# aid_adw.save_candidates(adw_files['candidates'])
aid_adw.load_candidates(adw_files['candidates'])
# aid_adw.compute_features(features='filtered')
# aid_adw.save_features(adw_files['feature_values'])
aid_adw.load_features(adw_files['feature_values'])
aid_adw.load_y_true(adw_files['y_true'])

aid_adw.load_predictors('best_predictors_all.pkl')
aid_adw.score()
aid_adw.save_scores('scores_all.pkl')
