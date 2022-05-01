How to use the ``AaLearner`` class to find a good augmentation for an image dataset
###################################################################################



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
    :caption: defining the training and validation datasets

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


.. code-block::
    :caption: defining our child network architecture

    child_network_architecture = cn.LeNet
    # or
    # child_network_architecture = cn.LeNet()
    # or 
    # child_network_architecture = lambda _ : cn.LeNet()


.. Note:: 
    
    The ``child_network_architecture`` parameter can either a ``nn.Module``
    instance, a ``type`` which inherits ``nn.Module``, or a ``function`` 
    which returns a ``nn.Module``.
    
    A downside of doing the first of the three is that the same randomly 
    initialized weights are used for every policy, whereas for the latter 
    two, the network weights are reinitialized before training each policy.


Defining our auto-augmentation learner
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There's quite a lot of configurable hyperparameters for the :class:`AaLearner`'s
but they can be divided into three categories: search space hyperparameters, 
child network training hyperparameters, and learner-specific hyperparameters.



.. code-block::

    # learner = aal.EvoLearner()
    # learner = aal.GenLearner()
    # learner = aal.GruLearner()
    # learner = aal.UcbLearner()
    learner = aal.RsLearner(
                            sp_num=7,
                            toy_size=0.01,
                            batch_size=4,
                            learning_rate=0.05,
                            max_epochs=float('inf'),
                            early_stop_num=35,
                            )
    learner.learn(train_dataset,
                val_dataset,
                child_network_architecture=child_network_architecture,
                iterations=15000)

.. Note:: 
    
    The ``child_network_architecture`` parameter can either a ``nn.Module``
    instance, a ``type`` which inherits ``nn.Module``, or a ``function`` 
    which returns a ``nn.Module``.
    
    A downside of doing the first of the three is that the same randomly 
    initialized weights are used for every policy, whereas for the latter 
    two, the network weights are reinitialized before training each policy.


You can set further hyperparameters when defining a AaLearner. 

Also, depending on what learner you are using, there might be unique hyperparameters.
For example, in the GRU learner you can tune the exploration parameter ``alpha``.

Viewing the results:

``.history`` is a list containing all the policies tested and the respective
accuracies obtained when trained using them.

.. code-block::
    
    print(learner.history)