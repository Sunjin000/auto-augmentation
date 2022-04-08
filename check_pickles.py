import pickle
from pprint import pprint

with open('randomsearch_logs.pkl', 'rb') as file:
    list = pickle.load(file)

print(len(list))

with open('gru_logs.pkl','rb') as file:
    list = pickle.load(file)

print(len(list))
