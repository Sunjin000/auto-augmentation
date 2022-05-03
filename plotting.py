#%%

import matplotlib.pyplot as plt
import numpy as np

#%%

# FASION
gen_learner = [0.8870999813079834, 0.8906000256538391, 0.8853999972343445, 0.8866000175476074, 0.8924000263214111, 0.8889999985694885, 0.8859999775886536, 0.8910999894142151, 0.8871999979019165, 0.8848000168800354]
rand_learner = [0.6222999691963196, 0.6868000030517578, 0.8374999761581421, 0.8370999693870544, 0.6934999823570251, 0.42819997668266296, 0.8423999547958374, 0.8331999778747559, 0.8079999685287476, 0.5971999764442444]
evo_leaner =  [0.828000009059906, 0.8159999847412109, 0.7329999804496765, 0.7329999804496765, 0.8119999766349792, 0.675000011920929, 0.7929999828338623, 0.5680000185966492, 0.7829999923706055, 0.7910000085830688]
gru_learner = [0.7490999698638916, 0.8359999656677246, 0.8394999504089355, 0.8366999626159668, 0.6847000122070312, 0.7816999554634094, 0.7787999510765076, 0.8385999798774719]
ucb_learner = []
baseline_fash = [0.7886000275611877, 0.7886000275611877, 0.7886000275611877, 0.7886000275611877, 0.7886000275611877, 0.7886000275611877, 0.7886000275611877, 0.7886000275611877, 0.7886000275611877, 0.7886000275611877]


# fig, ax = plt.subplots()
# ax.bar('Baseline', np.mean(baseline_fash), yerr=np.std(baseline_fash),capsize=10, color='teal')
# ax.bar('Genetic Learner', np.mean(gen_learner), yerr=np.std(gen_learner),capsize=10, color='teal')
plt.figure(figsize=(9, 5))

plt.plot('Baseline', np.mean(baseline_fash), 'o', color = 'teal')
plt.errorbar(x='Baseline', y=np.mean(baseline_fash), yerr=np.std(baseline_fash), capsize = 3, color = 'teal')

plt.plot('Genetic', np.mean(gen_learner), 'o', color = 'teal')
plt.errorbar(x='Genetic', y=np.mean(gen_learner), yerr=np.std(gen_learner), capsize = 3, color = 'teal')

plt.plot('Evo', np.mean(evo_leaner), 'o', color = 'teal')
plt.errorbar(x='Evo', y=np.mean(evo_leaner), yerr=np.std(evo_leaner), capsize = 3, color = 'teal')

plt.plot('GRU', np.mean(gru_learner), 'o', color = 'teal')
plt.errorbar(x='GRU', y=np.mean(gru_learner), yerr=np.std(gru_learner), capsize = 3, color = 'teal')

plt.plot('Rand', np.mean(rand_learner), 'o', color = 'teal')
plt.errorbar(x='Rand', y=np.mean(rand_learner), yerr=np.std(rand_learner), capsize = 3, color = 'teal')



plt.ylabel('Child network accuracy', fontsize=16)
plt.xlabel('Learner', fontsize=16)

#%%



# CIFAR
evo_cif = [0.6046000123023987, 0.6050999760627747, 0.5861999988555908, 0.5936999917030334, 0.5949000120162964, 0.5791000127792358, 0.6000999808311462, 0.6017000079154968, 0.5983999967575073, 0.5885999798774719]
gru_cif = [0.6056999564170837, 0.6329999566078186, 0.6171000003814697, 0.62909996509552, 0.6380999684333801, 0.6105999946594238, 0.6304000020027161, 0.6299999952316284, 0.6108999848365784]
rand_cif = [0.6085000038146973, 0.6218000054359436, 0.6029999852180481, 0.6187999844551086, 0.6155999898910522, 0.6212999820709229, 0.5877000093460083, 0.606499969959259, 0.5968999862670898, 0.5967000126838684]
gen_cif = [0.5967000126838684, 0.6065999865531921, 0.5791000127792358, 0.564300000667572, 0.5875999927520752, 0.5774999856948853, 0.6098999977111816, 0.5716999769210815, 0.5939000248908997, 0.6014999747276306]
baseline_cif = [0.5248000025749207, 0.5248000025749207, 0.5248000025749207, 0.5248000025749207, 0.5339000225067139, 0.5248000025749207, 0.5248000025749207, 0.5248000025749207, 0.5248000025749207, 0.5248000025749207,\
                0.5422000288963318, 0.5422000288963318, 0.5422000288963318, 0.5422000288963318, 0.5422000288963318, 0.5422000288963318, 0.5422000288963318, 0.5422000288963318, 0.5422000288963318, 0.5422000288963318]
ucb_cif = []
# %%


plt.figure(figsize=(9, 5))

plt.plot('Baseline', np.mean(baseline_cif), 'o', color = 'teal')
plt.errorbar(x='Baseline', y=np.mean(baseline_cif), yerr=np.std(baseline_cif), capsize = 3, color = 'teal')

plt.plot('Genetic', np.mean(gen_cif), 'o', color = 'teal')
plt.errorbar(x='Genetic', y=np.mean(gen_cif), yerr=np.std(gen_cif), capsize = 3, color = 'teal')

plt.plot('Evo', np.mean(evo_cif), 'o', color = 'teal')
plt.errorbar(x='Evo', y=np.mean(evo_cif), yerr=np.std(evo_cif), capsize = 3, color = 'teal')

plt.plot('GRU', np.mean(gru_cif), 'o', color = 'teal')
plt.errorbar(x='GRU', y=np.mean(gru_cif), yerr=np.std(gru_cif), capsize = 3, color = 'teal')

plt.plot('Rand', np.mean(rand_cif), 'o', color = 'teal')
plt.errorbar(x='Rand', y=np.mean(rand_cif), yerr=np.std(rand_cif), capsize = 3, color = 'teal')



plt.ylabel('Child network accuracy', fontsize=16)
plt.xlabel('Learner', fontsize=16)
# %%
