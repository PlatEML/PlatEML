import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population
from pymoo.util.misc import has_feasible


class RM_MEDA(NSGA2):
    def __init__(self, k=5, dynamic=False, **kwargs):
        """
        Regularity Model-Based Multiobjective Estimation of Distribution Algorithm (RM-MEDA)

        Parameters
        ----------
        n_offsprings : int
            The number of individuals created in each iteration.
        pop_size : int
            The number of individuals which are surviving from the offspring population (non-elitist)
        k: int
            Parameter of the rm_meda algorithm number of cluster
        """

        super().__init__(**kwargs)

        self._K = k
        self.dynamic = dynamic

    def _infill(self):
        pop, len_pop, len_off = self.pop, self.pop_size, self.n_offsprings
        xl, xu = self.problem.bounds()
        X = pop.get("X")
        Xp = RMMEDA_operator(X, self._K, self.problem.n_obj, xl, xu)

        # create the population to proceed further
        off = Population.new(X=Xp)
        return off

    def _advance(self, infills=None, **kwargs):
        if infills is not None:
            self.pop = Population.merge(self.pop, infills)
            self.evaluator.eval(
                self.problem,
                self.pop,
                t=self.n_gen,
                skip_already_evaluated=self.eliminate_duplicates,
            )
        # execute the survival to find the fittest solutions
        self.pop = self.survival.do(
            self.problem, self.pop, n_survive=self.pop_size, algorithm=self
        )

    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            self.opt = self.pop[self.pop.get("rank") == 0]


def RMMEDA_operator(PopDec, K, M, XLow, XUpp):
    N, D = PopDec.shape
    ## Modeling
    Model, probability = LocalPCA(PopDec, M, K)
    ## Reproduction
    OffspringDec = np.zeros((N, D))
    # Generate new trial solutions one by one
    for i in np.arange(N):
        # Select one cluster by Roulette-wheel selection
        k = (np.where(np.random.rand() <= probability))[0][0]
        # Generate one offspring
        if not len(Model[k]["eVector"]) == 0:
            lower = Model[k]["a"] - 0.25 * (Model[k]["b"] - Model[k]["a"])
            upper = Model[k]["b"] + 0.25 * (Model[k]["b"] - Model[k]["a"])
            trial = np.random.uniform(0, 1) * (upper - lower) + lower  # ,(1,M-1)
            sigma = np.sum(np.abs(Model[k]["eValue"][M - 1 : D])) / (D - M + 1)
            OffspringDec[i, :] = (
                Model[k]["mean"]
                + trial * Model[k]["eVector"][:, : M - 1].conj().transpose()
                + np.random.randn(D) * np.sqrt(sigma)
            )
        else:
            OffspringDec[i, :] = Model[k]["mean"] + np.random.randn(D)
        NN, D = OffspringDec.shape
        low = np.tile(XLow, (NN, 1))
        upp = np.tile(XUpp, (NN, 1))
        lbnd = OffspringDec <= low
        ubnd = OffspringDec >= upp
        OffspringDec[lbnd] = 0.5 * (PopDec[lbnd] + low[lbnd])
        OffspringDec[ubnd] = 0.5 * (PopDec[ubnd] + upp[ubnd])

    return OffspringDec


def LocalPCA(PopDec, M, K, max_iter=50):
    N, D = np.shape(PopDec)  # Dimensions
    Model = [
        {
            "mean": PopDec[k],  # The mean of the model
            "PI": np.eye(D),  # The matrix PI
            "eVector": [],  # The eigenvectors
            "eValue": [],  # The eigenvalues
            "a": [],  # The lower bound of the projections
            "b": [],
        }
        for k in range(K)
    ]  # The upper bound of the projections

    # Modeling
    for iteration in range(1, max_iter):
        # Calculate the distance between each solution and its projection in
        # affine principal subspace of each cluster
        distance = np.zeros((N, K))  # matrix of zeros N*K
        for k in range(K):
            distance[:, k] = np.sum(
                (PopDec - np.tile(Model[k]["mean"], (N, 1))).dot(Model[k]["PI"])
                * (PopDec - np.tile(Model[k]["mean"], (N, 1))),
                1,
            )
        # Partition
        partition = np.argmin(distance, 1)  # get the index of mins
        # Update the model of each cluster
        updated = np.zeros(K, dtype=bool)  # array of k false
        for k in range(K):
            oldMean = Model[k]["mean"]
            current = partition == k
            if sum(current) < 2:
                if not any(current):
                    current = [np.random.randint(N)]
                Model[k]["mean"] = PopDec[current, :]
                Model[k]["PI"] = np.eye(D)
                Model[k]["eVector"] = []
                Model[k]["eValue"] = []
            else:
                Model[k]["mean"] = np.mean(PopDec[current, :], 0)
                cc = np.cov(
                    (
                        PopDec[current, :]
                        - np.tile(Model[k]["mean"], (np.sum(current), 1))
                    ).T
                )
                eValue, eVector = np.linalg.eig(cc)
                rank = np.argsort(-(eValue), axis=0)
                eValue = -np.sort(-(eValue), axis=0)
                Model[k]["eValue"] = np.real(eValue).copy()
                Model[k]["eVector"] = np.real(eVector[:, rank]).copy()
                Model[k]["PI"] = Model[k]["eVector"][:, (M - 1) :].dot(
                    Model[k]["eVector"][:, (M - 1) :].conj().transpose()
                )

            updated[k] = (
                not any(current)
                or np.sqrt(np.sum((oldMean - Model[k]["mean"]) ** 2)) > 1e-5
            )

        # Break if no change is made
        if not any(updated):
            break

    # Calculate the smallest hyper-rectangle of each model
    for k in range(K):
        if len(Model[k]["eVector"]) != 0:
            hyperRectangle = (
                PopDec[partition == k, :]
                - np.tile(Model[k]["mean"], (sum(partition == k), 1))
            ).dot(Model[k]["eVector"][:, 0 : M - 1])
            Model[k]["a"] = np.min(hyperRectangle)  # this should by tested
            Model[k]["b"] = np.max(hyperRectangle)  # this should by tested
        else:
            Model[k]["a"] = np.zeros((1, M - 1))
            Model[k]["b"] = np.zeros((1, M - 1))

    # Calculate the probability of each cluster for reproduction
    # Calculate the volume of each cluster
    volume = np.array([Model[k]["b"] for k in range(K)]) - np.array(
        [Model[k]["a"] for k in range(K)]
    )  # this should be tested
    #    volume = prod(cat(1,Model.b)-cat(1,Model.a),2)
    # Calculate the cumulative probability of each cluster
    probability = np.cumsum(volume / np.sum(volume))

    return Model, probability
