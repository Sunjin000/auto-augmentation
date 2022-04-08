import matplotlib.pyplot as plt
import numpy as np


# Fixed seed (same as benchmark)

# Looking at last generation can make out general trends of which transformations lead to the largest accuracies


gen_1_acc = [0.1998, 0.1405, 0.1678, 0.9690, 0.9672, 0.9540, 0.9047, 0.9730, 0.2060, 0.9260, 0.8035, 0.9715, 0.9737, 0.14, 0.9645]

gen_2_acc = [0.9218, 0.9753, 0.9758, 0.1088, 0.9710, 0.1655, 0.9735, 0.9655, 0.9740, 0.9377]

gen_3_acc = [0.1445, 0.9740, 0.9643, 0.9750, 0.9492, 0.9693, 0.1262, 0.9660, 0.9760, 0.9697]

gen_4_acc = [0.9697, 0.1238, 0.9613, 0.9737, 0.9603, 0.8620, 0.9712, 0.9617, 0.9737, 0.1855]

gen_5_acc = [0.6445, 0.9705, 0.9668, 0.9765, 0.1142, 0.9780, 0.9700, 0.2120, 0.9555, 0.9732]

gen_6_acc = [0.9710, 0.9665, 0.2077, 0.9535, 0.9765, 0.9712, 0.9697, 0.2145, 0.9523, 0.9718, 0.9718, 0.9718, 0.2180, 0.9622, 0.9785]

gen_acc = [gen_1_acc, gen_2_acc, gen_3_acc, gen_4_acc, gen_5_acc, gen_6_acc]

gen_acc_means = []
gen_acc_stds = []

for val in gen_acc:
    gen_acc_means.append(np.mean(val))
    gen_acc_stds.append(np.std(val))



# Vary seed

gen_1_vary = [0.1998, 0.9707, 0.9715, 0.9657, 0.8347, 0.9655, 0.1870, 0.0983, 0.3750, 0.9765, 0.9712, 0.9705, 0.9635, 0.9718, 0.1170]

gen_2_vary = [0.9758, 0.9607, 0.9597, 0.9753, 0.1165, 0.1503, 0.9747, 0.1725, 0.9645, 0.2290]

gen_3_vary = [0.1357, 0.9725, 0.1708, 0.9607, 0.2132, 0.9730, 0.9743, 0.9690, 0.0850, 0.9755]

gen_4_vary = [0.9722, 0.9760, 0.9697, 0.1155, 0.9715, 0.9688, 0.1785, 0.9745, 0.2362, 0.9765]

gen_5_vary = [0.9705, 0.2280, 0.9745, 0.1875, 0.9735, 0.9735, 0.9720, 0.9678, 0.9770, 0.1155]

gen_6_vary = [0.9685, 0.9730, 0.9735, 0.9760, 0.1495, 0.9707, 0.9700, 0.9747, 0.9750, 0.1155, 0.9732, 0.9745, 0.9758, 0.9768, 0.1155]

gen_vary = [gen_1_vary, gen_2_vary, gen_3_vary, gen_4_vary, gen_5_vary, gen_6_vary]

gen_vary_means = []
gen_vary_stds = []

for val in gen_vary:
    gen_vary_means.append(np.mean(val))
    gen_vary_stds.append(np.std(val))





# Multiple runs 

gen_1_mult = [0.1762, 0.9575, 0.1200, 0.9660, 0.9650, 0.9570, 0.9745, 0.9700, 0.15, 0.23, 0.16, 0.186, 0.9640, 0.9650]

gen_2_mult = [0.17, 0.1515, 0.1700, 0.9625, 0.9630, 0.9732, 0.9680, 0.9633, 0.9530, 0.9640]

gen_3_mult = [0.9750, 0.9720, 0.9655, 0.9530, 0.9623, 0.9730, 0.9748, 0.9625, 0.9716, 0.9672]

gen_4_mult = [0.9724, 0.9755, 0.9657, 0.9718, 0.9690, 0.9735, 0.9715, 0.9300, 0.9725, 0.9695]

gen_5_mult = [0.9560, 0.9750, 0.8750, 0.9717, 0.9731, 0.9741, 0.9747, 0.9726, 0.9729, 0.9727]

gen_6_mult = [0.9730, 0.9740, 0.9715, 0.9755, 0.9761, 0.9700, 0.9755, 0.9750, 0.9726, 0.9748, 0.9705, 0.9745, 0.9752, 0.9740, 0.9744]



gen_mult = [gen_1_mult, gen_2_mult, gen_3_mult,  gen_4_mult, gen_5_mult, gen_6_mult]

gen_mult_means = []
gen_mult_stds = []

for val in gen_mult:
    gen_mult_means.append(np.mean(val))
    gen_mult_stds.append(np.std(val))

num_gen = [i for i in range(len(gen_mult))]


# Baseline
baseline = [0.7990 for i in range(len(gen_mult))]



# plt.errorbar(num_gen, gen_acc_means, yerr=gen_acc_stds, linestyle = 'dotted', label = 'Fixed seed GA')
# plt.errorbar(num_gen, gen_vary_means, linestyle = 'dotted', yerr=gen_vary_stds, label = 'Varying seed GA')
# plt.errorbar(num_gen, gen_mult_means, linestyle = 'dotted', yerr=gen_mult_stds, label = 'Varying seed GA 2')

plt.plot(num_gen, gen_acc_means, linestyle = 'dotted', label = 'Fixed seed GA')
plt.plot(num_gen, gen_vary_means, linestyle = 'dotted',  label = 'Varying seed GA')
plt.plot(num_gen, gen_mult_means, linestyle = 'dotted', label = 'Varying seed GA 2')

plt.plot(num_gen, baseline, label = 'Fixed seed baseline')


plt.xlabel('Generation', fontsize = 16)
plt.ylabel('Validation Accuracy', fontsize = 16)

plt.legend()

plt.savefig('GA_results.png')