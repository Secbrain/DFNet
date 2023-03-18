import joblib
import json
import numpy as np
import os
import pprint
from sklearn.ensemble import RandomForestClassifier
import sys

from data_utils import load_train_and_test_data
from metrics import tree_apl
from visualize_utils import visualize_decision_tree

def pre_order_tree(tree, node_id, current_rule, class_names, feature_names, leaf_list):
    # leaf node children_left == children_right == -1
    is_split_node = tree.children_left[node_id] != tree.children_right[node_id]
    if is_split_node:
        pre_order_tree(tree, tree.children_left[node_id], 
                       current_rule + ', ' + feature_names[tree.feature[node_id]] + ' <= ' + str(tree.threshold[node_id]),
                       class_names, feature_names, leaf_list)
        pre_order_tree(tree, tree.children_right[node_id], 
                       current_rule + ', ' + feature_names[tree.feature[node_id]] + ' > ' + str(tree.threshold[node_id]),
                       class_names, feature_names, leaf_list)
    else:
        class_id = np.argmax(tree.value[node_id][0])
        leaf_list.append(current_rule.strip(', ') + ' => ' + str(tree.value[node_id][0]) + class_names[class_id])

def merge_tree(tree, node_id):
    # leaf node children_left == children_right == -1
    is_split_node = tree.children_left[node_id] != tree.children_right[node_id]
    if is_split_node:
        left_class_id = merge_tree(tree, tree.children_left[node_id])
        right_class_id = merge_tree(tree, tree.children_right[node_id])
        if left_class_id == -1 or right_class_id == -1:
            return -1
        if left_class_id == right_class_id:
            tree.children_left[node_id] = -1
            tree.children_right[node_id] = -1
            return left_class_id
        else:
            return -1
    else:
        class_id = np.argmax(tree.value[node_id][0])
        return class_id

def rule_to_dpdk(result_save_path, rule_save_name, dpdk_save_name):
    template_begin = 'pipeline PIPELINE0 table 1 rule add match acl priority 1 raw 1 '
    template_end = '0 255 0 255 action fwd port 0 meter tc0 meter 0 policer g g y y r r'

    rule_set = np.load(os.path.join(result_save_path, rule_save_name))
    rule_num, pre_n_bytes, _ = rule_set.shape

    with open(os.path.join(result_save_path, dpdk_save_name), 'w+') as f:
        for i in range(rule_num):
            f.write(template_begin)
            for j in range(pre_n_bytes):
                if j == 0:
                    continue
                f.write(str(rule_set[i][j][0]) + ' ')
                f.write(str(rule_set[i][j][1]) + ' ')
            f.write(template_end)
            f.write('\n')

def tree_to_rule(tree, class_names, pre_n_bytes, result_save_path, visualize_save_name, rule_save_name, dpdk_save_name, train_data_path, class_num, train_data_num, random_state, test_data_path):
    # should input decision tree
    
    # leaf num
    leaf_list = []
    pre_order_tree(tree.tree_, 0, '', class_names, [str(i) for i in range(pre_n_bytes)], leaf_list)
    print('leaf num:', len(leaf_list))

    # merge num
    merge_tree(tree.tree_, 0)
    leaf_list = []
    pre_order_tree(tree.tree_, 0, '', class_names, [str(i) for i in range(pre_n_bytes)], leaf_list)
    print('merge num:', len(leaf_list))

    # merge visualize
    visualize_decision_tree(tree, pre_n_bytes, class_names, result_save_path, visualize_save_name)

    # rule num
    leaf_list = [i for i in leaf_list if i.endswith('Benign')]
    print('rule num:', len(leaf_list))
    pprint.pprint(leaf_list)

    # apl
    X_train, y_train, X_test, y_test = load_train_and_test_data(train_data_path, class_num, 
                                                                train_data_num, random_state, 
                                                                test_data_path, pre_n_bytes,
                                                                )
    apl = tree_apl(tree, X_test.cpu().numpy())
    print('APL: {0:.4f}'.format(apl))

    # save rules
    rule_set = np.zeros([len(leaf_list), pre_n_bytes, 2], dtype=np.int)
    rule_set[:, :, 1] = 255

    for i, rule in enumerate(leaf_list):
        condition_list = rule.split('=>')[0].split(',')
        for condition in condition_list:
            condition = condition.strip(' ')
            if '<=' in condition:
                byte, threshold = condition.split('<=')
                byte = byte.strip(' ')
                threshold = threshold.strip(' ')
                byte = int(byte)
                threshold = int(float(threshold))
                rule_set[i, byte, 1] = min(rule_set[i, byte, 1], threshold)
            elif '>' in condition:
                byte, threshold = condition.split('>')
                byte = byte.strip(' ')
                threshold = threshold.strip(' ')
                byte = int(byte)
                threshold = int(float(threshold)) + 1
                rule_set[i, byte, 0] = max(rule_set[i, byte, 0], threshold)

    np.save(os.path.join(result_save_path, rule_save_name), rule_set)

    rule_to_dpdk(result_save_path, rule_save_name, dpdk_save_name)


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
class_names = config['class_names']
result_save_path = config['result_save_path']

if not os.path.exists(result_save_path):
    print('[ERROR] Please train the model first!')
    exit(1)
print('[SUCCESS] Load and parse config\n\n')


print('[DOING] Tree to rule')
tree_file_list = sorted(os.listdir(result_save_path))
tree_file_list = [i for i in tree_file_list if i.endswith('.joblib')]

for tree_file_name in tree_file_list:
    print('\ntree name:', tree_file_name)
    tree = joblib.load(os.path.join(result_save_path, tree_file_name))
    if isinstance(tree, RandomForestClassifier):
        for idx, t in enumerate(tree.estimators_):
            print('\nTree:', idx)
            tree_to_rule(t, class_names, pre_n_bytes, result_save_path, 'visualize_tree_merge' + str(idx) + '.pdf', 'rule_set' + str(idx) + '.npy', 'rules' + str(idx) + '.txt', train_data_path, class_num, train_data_num, random_state, test_data_path)
    else:
        tree_to_rule(tree, class_names, pre_n_bytes, result_save_path, 'visualize_tree_merge.pdf', 'rule_set.npy', 'rules.txt', train_data_path, class_num, train_data_num, random_state, test_data_path)
print('[SUCCESS] Tree to rule')