import pickle 

with open('genetic_logs.pkl', 'rb') as f:
    y = pickle.load(f)

accs = [x[1] for x in y]

print("accs: ", accs)
print("len accs: ", len(accs))

with open("genetic_accs.txt", "w") as output:
    output.write(str(accs))