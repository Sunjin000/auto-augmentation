How-to guides
---------------------------------

This is a page dedicated to demonstrating functionalities of :class:`aa_learner`.

It is a how-to guide. (Using the terminology of https://documentation.divio.com/structure/)

###################################################
Using an AutoAutgment learner to find a good policy
###################################################

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

    We need the ``child_network`` variable to be a ``type``, not a ``nn.Module``
    because the ``child_network`` will be called multiple times to initialize a 
    ``nn.Module`` of its architecture multiple times: once every time we need to 
    train a different network to evaluate a different policy.



Using the random search learner to evaluate randomly generated policies:

.. code-block::

    rs_agent = aal.randomsearch_learner()
    rs_agent.learn(train_dataset, test_dataset, child_network, toy_flag=True)



Viewing the results:

``.history`` is a list containing all the policies tested and the respective
accuracies obtained when trained using them.

.. code-block::
    
    print(rs_agent.history)