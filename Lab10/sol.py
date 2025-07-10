import numpy
import scipy
import sklearn
import sklearn.datasets
import sklearn.neural_network
import sklearn.svm

import bayesRisk # Laboratory 8

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def split_db_2to1(D, L, seed=0):

    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)


def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L

def load_iris_binary():
    D, L = load_iris()
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L

# Train a MLP with data DTR and labels LTR
# We allow for different number / size of hidden layers layerSizes, different values of the regularizer coefficient lamb, different activation function act and different solver
# The remaining parameters are set to the values given in the laboratory text or use the default library values
# The regularization coefficient lamb is mapped to the MLPClassifier parameter alpha by alpha = lamb*DTR.shape[1]
# The function returns the neuralnetwork model and the empirical class prior, which is required to map the network outputs to socres that behave like class-condition log-likelihoods (up to an irrelevant constant, multiclass classification) or log-likelihood ratios (binary tasks)
def trainMLP(DTR, LTR, layerSizes, lamb, act='tanh', solver='adam'):

    clf = sklearn.neural_network.MLPClassifier(random_state=0, activation=act, solver=solver, verbose=False, max_iter=2000, alpha=lamb*DTR.shape[1], hidden_layer_sizes=layerSizes)
    clf.fit(DTR.T, LTR)
    return (clf, [(LTR==i).sum() / float(LTR.size) for i in numpy.unique(LTR)])

# Given a model, compute log-posterior probabilities for data D
def computeMLPLogPosteriorProbabilities(model, D):

    q = model.predict_log_proba(D.T).T
    return q

# Given a model and the training set empircal prior, convert class posterior probabilities to a vector of scores that behaves like class-conditional log-likelihoods, up to an irrelevant constant (i.e., compensate for the empirical training prior)
def computeMLPLogLikelihood(model, empPriorTrain, D):
    post = computeMLPLogPosteriorProbabilities(model, D)
    return post - vcol(numpy.log(empPriorTrain))

# Given a model, compute class log-posterior ratios for a binary task
def binaryMLPLogPosteriorRatio(model, D):
    
    ll = computeMLPLogPosteriorProbabilities(model, D)
    return ll[1]-ll[0]

# Given a model and the training set empircal prior, compute a score that can be interpreted as a log-likelihood ratio (i.e., compensate for the empirical training prior)
def binaryMLPLogLikelihoodRatio(model, empPriorTrain, D):
    
    ll = computeMLPLogLikelihood(model, empPriorTrain, D)
    return ll[1]-ll[0]


if __name__ == '__main__':

    D, L = load_iris_binary()
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    for layerSize in [ [], [10], [10, 10] ]:
        for lamb in [1e-3, 1e-2, 1e-1]:
            model, empPriorTrain = trainMLP(DTR, LTR, layerSize, lamb, solver='lbfgs')

            print ('layerSize:', layerSize, '- lambda:', lamb)
            print ('\tLoss: %.6e' % model.loss_)
            SVALLLR = binaryMLPLogLikelihoodRatio(model, empPriorTrain, DVAL)
            print ('\tminDCF - pT = 0.5: %.4f' % bayesRisk.compute_minDCF_binary_fast(SVALLLR, LVAL, 0.5, 1.0, 1.0))
            print ('\tactDCF - pT = 0.5: %.4f' % bayesRisk.compute_actDCF_binary_fast(SVALLLR, LVAL, 0.5, 1.0, 1.0))
            SVAL = binaryMLPLogPosteriorRatio(model, DVAL)
            PVAL = (SVAL > 0) * 1
            print ('\tError rate (log-posterior <> 0): %.1f%%' % ( ( 1 - (PVAL == LVAL).sum() / float(LVAL.size) ) * 100 ) )

