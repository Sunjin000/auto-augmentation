How to use the ``AaLearner`` class to find a good augmentation for an image dataset
###################################################################################

.. currentmodule:: autoaug.autoaugment_learners

This is a page dedicated to demonstrating functionalities of 
:class:`~autoaug.autoaugment_learners.AaLearner`.


Necessary Imports
^^^^^^^^^^^^^^^^^

.. code-block::

    # auto augment learners
    import autoaug.autoaugment_learners as aal
    # example CNN classifiers for the purpose of this how-to guide
    import autoaug.child_networks as cn

    import torchvision.datasets as datasets
    import torchvision



Defining the problem setting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We need to specify the train dataset, validation dataset, and the child network 
(a convolutional neural network image classifier) for which we want to obtain an
optimal image augmentation policy for.

.. code-block::
    :caption: Defining the training and validation datasets.

    train_dataset = datasets.MNIST(
                            root='./autoaug/datasets/mnist/train',
                            train=True, 
                            download=True, 
                            transform=None
                            )
    val_dataset = datasets.MNIST(
                            root='./autoaug/datasets/mnist/test', 
                            train=False, 
                            download=True, 
                            transform=torchvision.transforms.ToTensor()
                            )


We have used a dataset provided by the ``torchvision`` package
above, but you can use whatever ``torchvision.datasets.VisionDataset`` you want.


.. code-block::
    :caption: Defining our child network architecture.

    child_network_architecture = cn.LeNet
    # or
    # child_network_architecture = cn.LeNet()
    # or 
    # child_network_architecture = lambda _ : cn.LeNet()


We have used a child network archietcture (LeNet) provided by our package
above, but you can use whatever architecture of ``nn.Module`` you want.


.. Note:: 
    
    The ``child_network_architecture`` parameter can either a ``nn.Module``
    instance, a ``type`` which inherits ``nn.Module``, or a ``function`` 
    which returns a ``nn.Module``.
    
    We did this to make our code more flexible.

    A downside of doing the first of the three is that the same randomly 
    initialized weights are used for every policy, whereas for the latter 
    two, the network weights are reinitialized before training each policy.



Defining our auto-augmentation learner
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There's quite a lot of configurable hyperparameters for the :class:`AaLearner`'s
but they can be divided into three categories: search space hyperparameters, 
child network training hyperparameters, and learner-specific hyperparameters.

The search space hyperparameters and child network training hyperparameters are 
shared by all :class:`AaLearner`'s. Let us choose some here:

.. code-block::
    :caption: Defining search space and child network hyperparameters.

    # All the parameters specified here are shared across
    # all AaLearner's

    search_space_hyp = {
            # number of subpolicies per policy
            sp_num=5, 
            # number of bins for probability
            p_bins=10, 
            # number of bins for magnitude
            m_bins=11, 
            # image operations to exclude from
            exclude_method=['Invert', 'Solarize']
            }
    child_network_hyp = {
            learning_rate=0.01,
            max_epochs=float('inf'),
            early_stop_num=15,
            batch_size=16,
            toy_size=1 # using a toy size of 1 means 
                       # we use the whole dataset
            }
            
.. important::
    
    Choosing a good set of child network hyperparameters is very 
    important for a good performance of the :class:`AaLearner`'s.

    Hence we recommend doing a hyperparameter search over the 
    configurable child network hyperparemters before using our
    :class:`AaLearner`'s. If this is somehow not possible, we 
    recommend using our :class:`UcbLearner` as it is most adept 
    at dealing with uncertainty of the accuracy obtained by training
    a child network based on an augmentation policy.


.. code-block::
    :caption: Initializing our learner.

    num_offsprings = 4 # a GenLearner specific hyperparameter
    learner = aal.GenLearner(
                            **search_space_hyp,
                            **child_network_hyp,
                            num_offsprings=num_offsprings
                            )



Training the learner
^^^^^^^^^^^^^^^^^^^^

The following is the simplest way to train a :class:`AaLearner`:

.. code-block::
    :caption: Simplest way to use an AaLearner.
    
    learner.learn(
            train_dataset=train_dataset, 
            test_dataset=val_dataset, 
            child_network_architecture=child_network_architecture, 
            iterations = 500)

However doing so is not recommended because checkpoints are not saved
during training. As automatic augmentation is a computationally costly 
process, learning will take a long time. Hence if the software crashes, 
all progress will be lost.

Therefore, we recommend something like the following:

.. code-block::
    :caption: A Python script with checkpoints installed.
    
    save_directory = './saved_learners/my_gen_learner.pkl'
    total_iter = 500

    if __name__=='__main__':
        try:
            # try to load agent
            with open(save_directory, 'rb') as f:
                agent = pickle.load(f, map_location=device)
        except FileNotFoundError:
            # if agent hasn't been saved yet, initialize the agent
            agent = GenLearner(
                            **search_space_hyp,
                            **child_network_hyp,
                            num_offsprings=num_offsprings
                            )

        # if history is not length total_iter yet(if total_iter
        # different policies haven't been tested yet), keep running
        while len(agent.history)<total_iter:
            print(f'{len(agent.history)} / {total_iter}')
            # run 1 iteration (test one new policy and update the learner)
            agent.learn(
                        train_dataset=train_dataset,
                        test=val_dataset,
                        child_network_architecture=child_network_architecture,
                        iterations=1
                        )
            # save agent every iteration
            with open(save_directory, 'wb+') as f:
                pickle.dump(agent, f)

        print('run_benchmark closing')


Viewing the results
^^^^^^^^^^^^^^^^^^^

There are several ways to view the what the learner has found.

- :attr:`AaLearner.history` is a list containing all the policies tested 
  and the respective accuracies obtained when trained using them.
- :meth:`AaLearner.get_n_best_policies` shows the top n policies that the 
  learner has tested.
- If you want to create a mega policy containing the top n policies the 
  learner has tested, you can use :meth:`AaLearner.get_mega_policy`.



Using the results
^^^^^^^^^^^^^^^^^

In order to apply the obtained policy on an image dataset, see
:ref:`this How-to<autoaugment howto>`.