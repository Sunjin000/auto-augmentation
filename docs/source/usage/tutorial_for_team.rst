aa_learner object and its children
------------------------------------------------------------------------------------------------

This is a page dedicated to demonstrating functionalities of :class:`aa_learner`.

This is a how-to guide (in the sense describe in https://documentation.divio.com/structure/).

######################################################################################################
How to use the ``aa_learner`` class to find an optimal policy for a dataset-child_network pair
######################################################################################################

This section can also be read as a ``.py`` file in ``./tutorials/how_use_aalearner.py``.


.. code-block::

    import MetaAugment.autoaugment_learners as aal
    import MetaAugment.child_networks as cn

    import torchvision.datasets as datasets
    import torchvision



Defining the problem setting:

.. code-block::

    train_dataset = datasets.MNIST(root='./MetaAugment/datasets/mnist/train',
                                    train=True, download=True, transform=None)
    test_dataset = datasets.MNIST(root='./MetaAugment/datasets/mnist/test', 
                            train=False, download=True, transform=torchvision.transforms.ToTensor())
    child_network = cn.lenet


.. warning:: 
    
    In earlier versions, we had to write ``child_network=cn.LeNet`` 
    and not ``child_network=cn.LeNet()``. But now we can do both. 
    Both types of objects can be input into ``aa_learner.learn()``.
    
    A downside (or maybe the upside??) of doing the latter is that 
    the same randomly initialized weights are used for every policy.

Using the random search learner to evaluate randomly generated policies: (You
can use any other learner in place of random search learner as well)

.. code-block::

    # aa_agent = aal.gru_learner()
    # aa_agent = aal.evo_learner()
    # aa_agent = aal.ucb_learner()
    # aa_agent = aal.ac_learner()
    aa_agent = aal.randomsearch_learner(
                                    sp_num=7,
                                    toy_flag=True,
                                    toy_size=0.01,
                                    batch_size=4,
                                    learning_rate=0.05,
                                    max_epochs=float('inf'),
                                    early_stop_num=35,
                                    )
    aa_agent.learn(train_dataset,
                test_dataset,
                child_network_architecture=child_network,
                iterations=15000)

You can set further hyperparameters when defining a aa_learner. 

Also, depending on what learner you are using, there might be unique hyperparameters.
For example, in the GRU learner you can tune the exploration parameter ``alpha``.

Viewing the results:

``.history`` is a list containing all the policies tested and the respective
accuracies obtained when trained using them.

.. code-block::
    
    print(aa_agent.history)