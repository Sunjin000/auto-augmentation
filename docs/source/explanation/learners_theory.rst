The Theory Behind the Learners
******************************

.. currentmodule:: autoaug.autoaugment_learners




Random Search Learner (:class:`RsLearner`)
##########################################

This learner is a purely randomised searcher, yet despite this it 
is a hard baseline to beat, as it is in many automatic hyperparameter 
tuning problem settings :cite:t:`randomsearch`. Each subpolicy is found by the following 
process; an augmentation is chosen from the available selection with 
an equal probability, followed by a probability from 0.0 to 1.0 in 
incrementations of 0.1, which, again, are equally likely. If the 
selected augmentation requires a magnitude, we use Python's random 
module to select a random integer between 0 and 9 inclusive, otherwise 
the magnitude is set to None. This is repeated to generate a complete 
subpolicy, and can be joined into a policy to be tested. The 
corresponding accuracy of the child network is then stored with the 
policy for logging.



Genetic Learner (:class:`GenLearner`)
#####################################

The genetic learner has similar elements to the Random Search 
learner, but uses information from previous sub-policies when generating 
new ones to more efficiently search for optimal augmentation parameters. 
The methodology for this learner is shown in :ref:`gen_pseudocode`;


In this algorithm each subpolicy is represented as a binary string. 
For example the subpolicy ((ShearX, 0.9, 3), (TranslateY, 0.1, 7)) 
would be represented as '000010010011001100010111' where each 
augmentation, probability, and magnitude is parsed as a 4-bit long 
digit. We then rank the effectiveness of a policy by the accuracy of 
the trained child network implementing it, and carry this by storing 
tuples of (policy, accuracy). We randomly select parents as previous 
policies, where the probability of selection is weighted by the 
corresponding accuracy. A child is then produced using the random 
crossover method, and this is taken as the next policy to test. In the
case that the binary number produced from this method did not 
correspond to an available augmentation, probability, or magnitude, 
or if edge cases were found (e.g. an augmentation with required 
magnitude None was assigned a float value), such cases were resolved 
with uniform random selection.


.. _gen_pseudocode:

.. figure:: /img/GenLearner_pseudo.jpg
   :alt: Genlutionary Learner (GenLearner) pseudocode
   :align: center
   :width: 700px

   GenLearner Pseudocode


Evolutionary Learner (:class:`EvoLearner`)
##########################################


The Evolutionary learner is similar in principle to the Genetic 
learner except that instead of the subpolicy being expressed as 
genes, the genes are expressed via the weights of a neural network. 
In this library :cite:t:`pygad` is used, which allows an easy 
implementation of such an evolutionary approach that is compatible 
with PyTorch networks. The algorithm is shown in :numref:`evo_pseudocode`;. 
For this learner a child network and controller are required. The 
architecture of the child network must be able to pass images and 
classify them, whereas the controller network must be able to take 
the same input images and output a policy. In this work, this output 
consisted of the number of functions used in the search, in addition 
to 2 nodes for the probability and magnitudes respectively, and then 
doubling this to account for a subpolicy consisting of pairs of 
transformations.



In order to calculate which augmentations are optimal, we 
consider the strength of pairs of augmentations, as well as the 
presence of those exact pairs in the network output. We first find 
the covariance of the outputs for the first and second augmentations, 
and chose the most strongly correlated pair of augmentations. We 
then returned to the batched network output to find all instances 
where these pairs of augmentations were chosen, and found the mean 
magnitude and probability from these. This could be repeated to 
generate multiple different subpolicies, at which point it was 
passed to the child network for training.


.. _evo_pseudocode:

.. figure:: /img/EvoLearner_pseudo.jpg
   :alt: Evolutionary Learner (EvoLearner) pseudocode
   :align: center
   :width: 700px

   EvoLearner Pseudocode



GRU Learner (:class:`GruLearner`)
#################################

The algorithm is shown in :numref:`gru_pseudocode`.
This model nearly identical to what was used in 
:cite:t:`autoaugment`, where the authors borrowed the model from this 
:cite:t:`neural_arch_search`. In their works an LSTM 
controller was used, which output a policy in the form of a 
length 10 sequence of vectors, each vector representing an 
operation. The LSTM controller was updated using proximal policy 
optimization (PPO) :cite:t:`ppo`, using the accuracy of the child 
network as the reward value. 

In the context of the PPO update, which was developed in the 
reinforcement learning literature, the subpolicies are the 
'actions' of our RL agent. 

We use GRU instead of LSTM because it's faster while empirically 
being functionally equivalent :cite:t:`gru`. 

For more details on how an RNN (LSTM or GRU) network can be 
updated using PPO see Section 3.2 of :cite:t:`neural_arch_search`. 
For more details on the hyperparameters of the LSTM they used during 
training, see 'Training Details' in the same paper.


.. _gru_pseudocode:

.. figure:: /img/GruLearner_pseudo.jpg
   :alt: GRU Learner (GruLearner) pseudocode
   :align: center
   :width: 700px

   GruLearner Pseudocode



UCB Learner (:class:`UcbLearner`)
#################################

UCB1 is often described as optimism in the face of uncertainty, 
and is described in :numref:`ucb_pseudocode`. We are trying to 
find the action with the highest reward given some uncertainty of 
the current expected rewards (:math:`q`-values). As the number of times an 
action has been taken  increases (relative to the total number of 
actions taken), its outcome becomes more certain. Instead of looking 
at raw :math:`q`-values, the algorithm takes the action counts into account, 
which can be seen in Equation :eq:`ucbeq`.


.. math::
    :label: ucbeq

    q_{i} + \sqrt{\frac{2ln(n)}{n_i}}


where :math:`i`` represents an action, :math:`q_i` is the associated :math:`q`-value, 
:math:`n` is the total action count, and :math:`n_i` is the action count. UCB1 
is a typical algorithm used in multi-arm bandit problems 
:cite:t:`kexugit_test_nodate` and, since the auto-augmentation problem 
can be rephrased as a multi-arm bandit problem (where each arm is 
applying an augmentation and each return is an accuracy score on a 
validation set), it is simple to use UCB1 as a starting case before 
moving to more complex algorithms. UCB1 is easy to understand and easy 
to implement but does not use neural networks, and so offers no real 
way to generalise between different types of augmentations.

It works by keeping track of how many times each augmentation has been 
attempted, and adjusting the raw :math:`q`-values (mean accuracy on validation 
set) by these counts (seen in the formula above). This new adjusted 
q-value is then used to decide which augmentation is applied in the next 
iteration (that is, which bandit arm is pulled).

Due to the fact UCB1 doesn't generalise between types of augmentation, 
it is quite slow compared to methods that involve neural networks (for 
example GRU). For this reason, we used a similar methodology to 
:cite:t:`cubuk_autoaugment_2019` however we limit the 
search space to 5 policies each with 5 subpolicies (the original paper 
has 14,000 policies each with 5 subpolicies). The raw accuracy on a 
fixed validation set is fed in as the reward.

UCB1 is very good at balancing the exploration-exploitation trade-off 
in an efficient manner that minimises regret, which is why it's often 
the go-to algorithm for multi-arm bandit problems.


.. _ucb_pseudocode:

.. figure:: /img/UcbLearner_pseudo.jpg
   :alt: UCB1 Learner (UcbLearner) pseudocode
   :align: center
   :width: 700px

   UcbLearner Pseudocode



.. bibliography::