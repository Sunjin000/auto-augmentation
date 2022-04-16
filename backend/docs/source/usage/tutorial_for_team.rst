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



.. note:: 
    It is important not to type

    .. code-block::

        child_network = cn.lenet()

    We need the ``child_network`` variable to be a ``type`` object, not a ``nn.Module`` object
    because the ``child_network`` will be called multiple times to initialize a 
    ``nn.Module`` of its architecture multiple times. This is because every time
    we need to evaluate a different policy, we need to train another new network
    of the same architecture.



Using the random search learner to evaluate randomly generated policies: (You
can use any other learner in place of random search learner as well)

.. code-block::

    # aa_agent = aal.gru_learner()
    # aa_agent = aal.evo_learner()
    # aa_agent = aal.ucb_learner()
    # aa_agent = aal.ac_learner()
    aa_agent = aal.randomsearch_learner()
    aa_agent.learn(train_dataset, test_dataset, child_network)



Viewing the results:

``.history`` is a list containing all the policies tested and the respective
accuracies obtained when trained using them.

.. code-block::
    
    print(aa_agent.history)