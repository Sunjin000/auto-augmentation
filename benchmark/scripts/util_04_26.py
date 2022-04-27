from cProfile import label
import torch
import matplotlib.pyplot as plt

def get_best_acc(
    save_file,
    ):
    """
    Use this to get the best accuracy history of the pickles 

    Args:
        save_file (str): pickle directory

    Returns:
        list[floats]: best_accuracy_list
    """
    # try to load agent
    with open(save_file, 'rb') as f:
        agent = torch.load(f)
        history = agent.history
    
    best_acc_list = []
    best_acc = 0.0

    for policy, acc in history:
        best_acc = max(best_acc, acc)
        best_acc_list.append(best_acc)

    return best_acc_list

plt.plot(get_best_acc('benchmark/pickles/04_22_cf_ln_gru.pkl'),
            label='GRU')
print('1 done')
plt.plot(get_best_acc('benchmark/pickles/04_22_cf_ln_rs.pkl'),
            label='RandomSearch')
print('2 done')
plt.xlabel('no. of child networks trained')
plt.ylabel('highest accuracy obtained until now')
plt.legend()
plt.show()


plt.plot(get_best_acc('benchmark/pickles/04_22_fm_sn_gru.pkl'),
            label='GRU')
print('3 done')
plt.plot(get_best_acc('benchmark/pickles/04_22_fm_sn_rs.pkl'),
            label='RandomSearch')
print('4 done')
plt.xlabel('no. of child networks trained')
plt.ylabel('highest accuracy obtained until now')
plt.legend()
plt.show()


print('FashionMNIST_GRU')
print(get_best_acc('benchmark/pickles/04_22_fm_sn_gru.pkl'))
print()
print('FashionMNIST_RandomSearch')

print(get_best_acc('benchmark/pickles/04_22_fm_sn_rs.pkl'))
print()

print('CIFAR_GRU')
print(get_best_acc('benchmark/pickles/04_22_cf_ln_gru.pkl'))
print()

print('CIFAR_RandomSearch')
print(get_best_acc('benchmark/pickles/04_22_cf_ln_rs.pkl'))
print()