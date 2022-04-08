import pickle
from pprint import pprint
import matplotlib.pyplot as plt
from torch import gru

def get_maxacc(log):
    output = []
    maxacc = 0
    for policy, acc in log:
        maxacc = max(maxacc, acc)
        output.append(maxacc)
    return output

with open('randomsearch_logs.pkl', 'rb') as file:
    rs_list = pickle.load(file)

with open('gru_logs.pkl', 'rb') as file:
    gru_list = pickle.load(file)


plt.plot(get_maxacc(rs_list), label='randomsearcher')
plt.plot(get_maxacc(gru_list), label='gru learner')
plt.title('Comparing two agents')
plt.ylabel('best accuracy to date')
plt.xlabel('number of policies tested')
plt.legend()
plt.show()

plt.plot([acc for pol,acc in rs_list], label='randomsearcher')
plt.plot([acc for pol,acc in gru_list], label='gru learner')
plt.title('Comparing two agents')
plt.ylabel('best accuracy to date')
plt.xlabel('number of policies tested')
plt.legend()
plt.show()


def get_best5(log):
    l = sorted(log, reverse=True, key=lambda x:x[1])
    return (l[:5])

def get_worst5(log):
    l = sorted(log, key=lambda x:x[1])
    return (l[:5])

pprint(get_best5(rs_list))
pprint(get_best5(gru_list))