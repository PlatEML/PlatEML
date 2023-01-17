"""
Feature selection tool based on MOEA algorithms
"""

import base64

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from sklearn.base import BaseEstimator
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectorMixin
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

from .rm_meda import RM_MEDA

translate = {
    "chinese":
        {
            "Feature_Selection_Program": "特征选择方案",
            "R2_Score": "R2分数",
        },
    "english":
        {
            "Feature_Selection_Program": "Feature Selection Program",
            "R2_Score": "R2 Score",
        }
}

class FeatureSelectionProblem(Problem):

    def __init__(self, X, y, model):
        super().__init__(n_var=X.shape[1], n_obj=2, n_constr=0, xl=0.0, xu=1.0)
        self.X = X
        self.y = y
        self.dim = X.shape[1]
        self.model: RandomForestRegressor = model

    def _evaluate(self, data, out, *args, **kwargs):
        all_data = []
        for x in data:
            x = np.round(x)
            if x.sum() < 1:
                all_data.append([1, 1])
                continue
            f1 = np.sum(x)
            f2 = -1 * np.nan_to_num(cross_val_score(self.model, self.X[:, x.astype(bool)], self.y))
            all_data.append([f1, np.mean(f2)])
        out["F"] = np.array(all_data)


from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_reference_directions
from pymoo.factory import get_termination
from pymoo.optimize import minimize


class MyCallback(Callback):

    def __init__(self, state) -> None:
        super().__init__()
        self.state = state

    def notify(self, algorithm: NSGA2):
        print(algorithm.termination.n_max_gen)
        self.state[0] = (algorithm.n_gen / algorithm.termination.n_max_gen) * 100


class MOEASelector(SelectorMixin, BaseEstimator):

    def __init__(self, estimator, selection_operator, pop_size=40, generation=10, state=None):
        self.pop_size = int(pop_size)
        self.generation = int(generation)
        self.estimator = estimator
        self.selection_operator = selection_operator
        self.state = state

    def _get_support_mask(self):
        pass

    def fit(self, X, y=None, **kwargs):
        self.problem = FeatureSelectionProblem(X, y, self.estimator)
        if self.selection_operator == 'NSGA2':
            algorithm = NSGA2(
                pop_size=self.pop_size,
                n_offsprings=self.pop_size,
                eliminate_duplicates=True
            )
        elif self.selection_operator == 'MOEA/D':
            ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=self.pop_size - 1)
            algorithm = MOEAD(
                ref_dirs,
                n_offsprings=self.pop_size,
            )
        elif self.selection_operator == 'NSGA3':
            ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=self.pop_size - 1)
            algorithm = NSGA3(
                ref_dirs,
                n_offsprings=self.pop_size,
                eliminate_duplicates=True
            )
        elif self.selection_operator == 'C-TAEA':
            ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=self.pop_size - 1)
            algorithm = CTAEA(
                ref_dirs,
                eliminate_duplicates=True
            )
        elif self.selection_operator == 'RM-MEDA':
            algorithm = RM_MEDA(
                pop_size=self.pop_size,
                n_offsprings=self.pop_size,
                eliminate_duplicates=True
            )
        else:
            raise Exception

        termination = get_termination("n_gen", self.generation)
        callback = MyCallback(self.state)
        res = minimize(self.problem,
                       algorithm,
                       termination,
                       seed=1,
                       callback=callback,
                       save_history=True,
                       verbose=True)

        X = res.X
        F = res.F
        F[:, 1] = -1 * F[:, 1]
        self.pf = F
        self.ps = X
        return X

    def plot_figure(self):
        plt.clf()
        sns.set_theme(style='white')
        obj = self.pf
        plt.scatter(obj[:, 0], obj[:, 1])
        plt.xlabel('Number of features')
        plt.ylabel('$R^2$ Score')
        # plt.show()
        # convert ot base64
        import io
        my_stringIObytes = io.BytesIO()
        plt.savefig(my_stringIObytes, format='jpg')
        my_stringIObytes.seek(0)
        my_base64_jpgData = base64.b64encode(my_stringIObytes.read())
        return my_base64_jpgData


def feature_selection_utils(pop_size, generation, global_data, operator_select, state, language):
    dt = DecisionTreeRegressor()
    x, y = np.array(global_data)[:, :-1], np.array(global_data)[:, -1]
    selector = MOEASelector(dt, operator_select, pop_size, generation, state)
    selector.fit(x, y)
    selection_result = pd.DataFrame(np.round(selector.ps)).astype(int)
    df = pd.DataFrame([
        selection_result.apply(lambda row: ','.join(row.values.astype(str)), axis=1),
        selector.pf[:, 1]
    ], index=[f"{translate[language]['Feature_Selection_Program']}", f"{translate[language]['R2_Score']}"]).T
    df = df.sort_values([f"{translate[language]['R2_Score']}"], ascending=False)
    return df, selector.plot_figure()


if __name__ == '__main__':
    dt = DecisionTreeRegressor()
    x, y = load_boston(return_X_y=True)
    # selector = MOEASelector(dt, 'RM-MEDA')
    # selector.fit(x, y)
    # print(selector.plot_figure())
    # print(feature_selection_utils(40, 10, np.concatenate([x, np.reshape(y, (-1, 1))], axis=1), 'NSGA3', [0]))

    # for operator in ['NSGA3', 'RM-MEDA', 'MOEA/D', 'NSGA2', 'C-TAEA']:
    for operator in ['C-TAEA']:
        print(feature_selection_utils(40, 10, np.concatenate([x, np.reshape(y, (-1, 1))], axis=1), operator, [0]))
