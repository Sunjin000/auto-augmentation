The Theory Behind the Learners
******************************

.. currentmodule:: autoaug.autoaugment_learners



Evolutionary Learner (:class:`EvoLearner`)
##########################################




Genetic Learner (:class:`GenLearner`)
#####################################




GRU Learner (:class:`GruLearner`)
#################################




Random Search Learner (:class:`RsLearner`)
##########################################




UCB Learner (:class:`UcbLearner`)
#################################


UCB1 is often described as optimism in the face of uncertainty. 
We are trying to find the action with the highest reward given some 
uncertainty of the current expected rewards (q-values). As the number
of times an action has been taken  increases (relative to the total 
number of actions taken), its outcome becomes more certain. Instead 
of looking at raw q-values, the algorithm takes the action counts 
into account using the following expression 
(where :math:`q_i` is the :math:`q`-value for action :math:`i`, :math:`n` 
is the total action count and :math:n_i is the action count for 
action :math:`i`):

.. math::
    q_i + \sqrt{\frac{2ln(n)}{n_i}}

UCB1 is a typical algorithm used in multi-arm bandit problems 
:cite:t:`kexugit_test_nodate`. Since the auto-augmentation problem 
can be rephrased as a multi-arm bandit problem (where each arm is 
applying an augmentation and each return is an accuracy score on a 
validation set), it is obvious to use UCB1 as a simple starting case 
before moving to more complex algorithms. UCB1 is easy to understand 
and easy to implement, however it doesn't use neural networks and so 
offers no real way to generalise between different types of augmentations.

It works by keeping track of how many times each augmentation has been 
attempted, and adjusting the raw q-values (mean accuracy on validation 
set) by these counts (seen in the formula above). This new adjusted 
q-value is then used to decide which augmentation is applied in the 
next iteration (that is, which bandit arm is pulled).

Due to the fact UCB1 doesn't generalise between types of augmentation, 
it is quite slow compared to methods that involve neural networks 
(for example GRU). For this reason, we used a similar methodology to 
the AutoAugment paper :cite:t:`cubuk_autoaugment_2019` however we limit 
the search space to 5 policies each with 5 subpolicies (the original 
paper has 14,000 policies each with 5 subpolicies). The raw accuracy 
on a fixed validation set is fed in as the reward.

UCB1 is very good at balancing the exploration-exploitation trade-off 
in an efficient manner that minimises regret, which is why it's often 
the go-to algorithm for multi-arm bandit problems.