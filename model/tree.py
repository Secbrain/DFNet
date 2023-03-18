import joblib
import json
import os
import pprint
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import sys

from data_utils import load_train_and_test_data
from metrics import print_and_ret_all_metrics
from visualize_utils import visualize_decision_tree


print('[DOING] Load and parse config')
# load config
# config_path = './config.json'
config_path = sys.argv[1]
with open(config_path) as f:
    config = json.load(f)
pprint.pprint(config)

# parse parameters
train_data_path = config['train_data_path']
test_data_path = config['test_data_path']
train_data_num = config['train_data_num']
class_num = config['class_num']
random_state = config['random_state']
pre_n_bytes = config['pre_n_bytes']
tree_model = config['tree_model']
tree_num = config['tree_num']
min_samples_leaf = config['min_samples_leaf']
result_save_path = config['result_save_path']
class_names = config['class_names']

if not os.path.exists(result_save_path):
    os.makedirs(result_save_path)
print('[SUCCESS] Load and parse config\n\n')


print('[DOING] Load data')
# load data
X_train, y_train, X_test, y_test = load_train_and_test_data(train_data_path, class_num, 
                                                            train_data_num, random_state, 
                                                            test_data_path, pre_n_bytes,
                                                            )
print('[SUCCESS] Load data\n\n')


print('[DOING] Train model:', tree_model)
# tree training
if tree_model == 'decisionTree':
    tree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
    tree.fit(X_train.cpu().numpy(), y_train.cpu().numpy())
    joblib.dump(tree, os.path.join(result_save_path, 'model_tree.joblib'))
elif tree_model == 'randomForest':
    tree = RandomForestClassifier(n_estimators=tree_num, min_samples_leaf=min_samples_leaf)
    tree.fit(X_train.cpu().numpy(), y_train.cpu().numpy())
    joblib.dump(tree, os.path.join(result_save_path, 'model_rf.joblib'))
print('[SUCCESS] Train model:', tree_model, '\n\n')


print('[DOING] Metrics')
# metrics
y_true = y_test.cpu().numpy()
y_pred = tree.predict(X_test.cpu().numpy())
print_and_ret_all_metrics(y_true, y_pred, X_test.cpu().numpy(), tree)
print('\nBenign 1 DDoS 0')
y_true_xor = y_true ^ 1
y_pred_xor = y_pred ^ 1
print_and_ret_all_metrics(y_true_xor, y_pred_xor, X_test.cpu().numpy(), tree)
print('[SUCCESS] Metrics\n\n')


print('[DOING] Visualize')
# visualize
if tree_model == 'decisionTree':
    visualize_decision_tree(tree, pre_n_bytes, class_names, result_save_path, 'visualize_tree.pdf')
elif tree_model == 'randomForest':
    for idx, t in enumerate(tree.estimators_):
        visualize_decision_tree(t, pre_n_bytes, class_names, result_save_path, 'visualize_tree' + str(idx) + '.pdf')
print('[SUCCESS] Visualize\n\n')