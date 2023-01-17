import base64
import io

import csv
import operator

import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from evolutionary_forest.forest import EvolutionaryForestRegressor
from gplearn.genetic import SymbolicRegressor
from pandas import DataFrame
from pstree.cluster_gp_sklearn import PSTreeRegressor, GPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_graphviz, plot_tree
from evolutionary_forest.utils import get_feature_importance, plot_feature_importance, feature_append
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

import collections
import pydotplus

operator_map = {
    '+': 'add',
    '-': 'sub',
    '*': 'mul',
    '/': 'div',
}

infix_map = {
    'ARG': 'X',
    'subtract': 'sub',
    'multiply': 'mul',
    'analytical_quotient': 'div',
}

translate = {
    "chinese":
        {
            "Data": "数据",
            "Num": "编号",
            "Predicted_Val": "预测值",
            "Actual_Val": "实际值",
            "Bias": "偏差",
        },
    "english":
        {
            "Data": "Data",
            "Num": "Num",
            "Predicted_Val": "Predicted Val",
            "Actual_Val": "Actual Val",
            "Bias": "Bias",
        }
}

class SymbolicRegressorPlus(SymbolicRegressor):

    def _verbose_reporter(self, run_details=None):
        super()._verbose_reporter(run_details)
        self.state[0] += 100 / self.generations


def ps_tree_predict(pop_size, gen, tournament_size, data, operator_select, state, language):
    gp = PSTreeRegressor(regr_class=GPRegressor, tree_class=DecisionTreeRegressor,
                         height_limit=6, n_pop=pop_size, n_gen=gen, normalize=False,
                         basic_primitive='optimal', size_objective=True,
                         survival_selection=operator_select)
    gp = Pipeline([
        ('scaler', StandardScaler()),
        ('gp', gp),
    ])
    state[0] = 50
    gp.fit(np.array(data.iloc[:, :-1]), data.iloc[:, -1])
    state[0] = 75
    real_value = data.iloc[:, -1]
    predict_value = np.round(gp.predict(np.array(data.iloc[:, :-1])), 3)
    result = DataFrame([np.arange(len(data)), predict_value,
                        real_value,
                        np.round(abs(real_value - predict_value), 3)], )
    result.columns = [f"{translate[language]['Data']}" + str(c) for c in result.columns]
    result.insert(0, '', [f"{translate[language]['Num']}", f"{translate[language]['Predicted_Val']}", f"{translate[language]['Actual_Val']}", f"{translate[language]['Bias']}"])

    # print(gp['gp'].tree)
    dot_data = export_graphviz(gp['gp'].tree, out_file=None, filled=True)
    dot_data = dot_data.replace('\n', '') #决策树右边有一个黑块，这是去掉那个黑块的
    #以下部分是改决策树节点颜色
    graph = pydotplus.graph_from_dot_data(dot_data)
    # graph = graphviz.Source(dot_data)
    colors = ('#90EE90', '#DDA0DD')
    edges = collections.defaultdict(list)
    for edge in graph.get_edge_list():
        edges[edge.get_source()].append(int(edge.get_destination()))
    for edge in edges:
        edges[edge].sort()
        for i in range(2):
            print(str(edges[edge][i]))
            dest = graph.get_node(str(edges[edge][i]))[0]
            dest.set_fillcolor(colors[i])

    print(graph)
    print(dir(graph))
    data = io.BytesIO()
    graph.write_png(data)
    # data.write(graph.pipe(format="png"))
    base64_data = base64.b64encode(data.getvalue())
    data.close()
    base64_datas = [
        base64_data
    ]
    train_infos = []
    state[0] = 100
    return result, base64_datas, train_infos, gp


def evolutionary_forest_predict(pop_size, gen, tournament_size, data, operator_select, state, language):
    gp = EvolutionaryForestRegressor(max_height=3, normalize=True, select='AutomaticLexicase',
                                     gene_num=10, boost_size=5, n_gen=gen, n_pop=pop_size, cross_pb=1,
                                     base_learner='Random-DT', verbose=True, max_tree_depth=3)
    state[0] = 50
    gp.fit(np.array(data.iloc[:, :-1]), np.array(data.iloc[:, -1]))
    state[0] = 75
    real_value = data.iloc[:, -1]
    predict_value = np.round(gp.predict(np.array(data.iloc[:, :-1])), 3)
    result = DataFrame([np.arange(len(data)), predict_value,
                        real_value,
                        np.round(abs(real_value - predict_value), 3)], )
    result.columns = [f"{translate[language]['Data']}" + str(c) for c in result.columns]
    result.insert(0, '', [f"{translate[language]['Num']}", f"{translate[language]['Predicted_Val']}", f"{translate[language]['Actual_Val']}", f"{translate[language]['Bias']}"])

    fig, axes = plt.subplots(nrows=1, ncols=min(5, len(gp.hof)), figsize=(30, 6))
    for index, h in enumerate(gp.hof):
        plot_tree(h.pipe['Ridge'], max_depth=0, filled=True, ax=axes[index])

    # 重要特征转Base-64
    my_stringIObytes = io.BytesIO()
    plt.savefig(my_stringIObytes, format='jpg')
    my_stringIObytes.seek(0)
    my_base64_jpgData = base64.b64encode(my_stringIObytes.read())

    base64_datas = [my_base64_jpgData]
    train_infos = []
    state[0] = 100

    return result, base64_datas, train_infos, gp


def gp_predict(pop_size, gen, tournament_size, data, operator_select, state, language):
    print('parameter', pop_size, gen)
    gp = SymbolicRegressorPlus(population_size=pop_size,
                               generations=gen,
                               tournament_size=tournament_size,
                               function_set=[operator_map[x] for x in operator_select],
                               verbose=True)
    gp.state = state
    gp.fit(data.iloc[:, :-1], data.iloc[:, -1])

    real_value = data.iloc[:, -1]
    predict_value = np.round(gp.predict(data.iloc[:, :-1]), 3)
    result = DataFrame([np.arange(len(data)), predict_value,
                        real_value,
                        np.round(abs(real_value - predict_value), 3)], )
    result.columns = [f"{translate[language]['Data']}" + str(c) for c in result.columns]
    result.insert(0, '', [f"{translate[language]['Num']}", f"{translate[language]['Predicted_Val']}", f"{translate[language]['Actual_Val']}", f"{translate[language]['Bias']}"])

    sorted_gp = list(sorted(gp._programs[-1], key=lambda x: x.fitness_))
    base64_datas = []
    train_infos = []

    # for tree in sorted_gp[:5]:
    for tree in sorted_gp:
        dot_data = tree.export_graphviz()
        graph = graphviz.Source(dot_data)
        data = io.BytesIO()
        data.write(graph.pipe(format="png"))
        base64_data = base64.b64encode(data.getvalue())
        base64_datas.append(base64_data)
        train_infos.append({
            'tree': str(tree),
            'mse': round(tree.fitness_, 3),
        })
        data.close()
    return result, base64_datas, train_infos, gp

def test_predict(data, gp, flag, language):
    preidict_value = []
    if flag == 1:
        predict_value = np.round(gp.predict(data), 3)
        result = DataFrame([np.arange(len(data)), predict_value])
        result.columns = [f"{translate[language]['Data']}" + str(c) for c in result.columns]
        result.insert(0, '', [f"{translate[language]['Num']}", f"{translate[language]['Predicted_Val']}"])
    else:
        predict_value = np.round(gp.predict(data.iloc[:, :-1]), 3)
        real_value = data.iloc[:, -1]
        result = DataFrame([np.arange(len(data)), predict_value,
                            real_value,
                            np.round(abs(real_value - predict_value), 3)], )
        result.columns = [f"{translate[language]['Data']}" + str(c) for c in result.columns]
        result.insert(0, '', [f"{translate[language]['Num']}", f"{translate[language]['Predicted_Val']}", f"{translate[language]['Actual_Val']}", f"{translate[language]['Bias']}"])
    return result

def scores_difference(x_train, x_test, new_train, new_test, y_train, y_test, session_id):
    regressor_list = ['RF', 'ET', 'AdaBoost', 'GBDT'] #, 'DART', 'XGBoost', 'LightGBM', 'CatBoost'
    regressor_dict = {
            'RF': RandomForestRegressor(n_estimators=100, n_jobs=-1),
            'ET': ExtraTreesRegressor(n_estimators=100, n_jobs=-1),
            'AdaBoost': AdaBoostRegressor(n_estimators=100),
            'GBDT': GradientBoostingRegressor(n_estimators=100),
            # 'DART': LGBMRegressor(n_jobs=1, n_estimators=100, boosting_type='dart',
            #                       xgboost_dart_mode=True),
            # 'XGBoost': XGBRegressor(n_jobs=1, n_estimators=100),
            # 'LightGBM': LGBMRegressor(n_jobs=1, n_estimators=100),
            # 'CatBoost': CatBoostRegressor(n_estimators=100, thread_count=1,
            #                               verbose=False, allow_writing_files=False),
        }
    scores_base = []
    scores_enhanced = []
    for regr in regressor_list:
        # Training on original features
        regressor = regressor_dict[regr]
        regressor.fit(x_train, y_train)
        score = r2_score(y_test, regressor.predict(x_test))
        base_score = (regr, score)
        scores_base.append(base_score)
        # Training on new features
        regressor.fit(new_train, y_train)
        score = r2_score(y_test, regressor.predict(new_test))
        enhanced_score = (regr, score)
        scores_enhanced.append(enhanced_score)

    base_frame = pd.DataFrame(scores_base, columns=['algorithm', 'score'])
    base_frame['score'] = pd.to_numeric(base_frame['score'])
    improved_frame = pd.DataFrame(scores_enhanced, columns=['algorithm', 'score'])
    improved_frame['score'] = pd.to_numeric(improved_frame['score'])
    base_score = base_frame.groupby(['algorithm']).mean()
    improved_score = improved_frame.groupby(['algorithm']).mean()

    sns.set(style="white", font_scale=1.5)
    width = 0.4
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(base_score.index, base_score['score'], width, label='Original Features')
    difference = improved_score['score'] - base_score['score']
    print(np.where(difference > 0, 'g', 'y'))
    ax.bar(base_score.index, difference, width, bottom=base_score['score'],
           label='Constructed Features',
           color=np.where(difference > 0, 'r', 'y'))
    ax.set_ylabel('Score ($R^2$)')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("temp/" + str(session_id) + ".png")


