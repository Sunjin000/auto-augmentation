AutoAugment object
------------------

######################################################################################################
How to use a ``AutoAugment`` object to apply AutoAugment policies to ``Datasets`` objects
######################################################################################################

This is a page dedicated to demonstrating functionalities of :class:`AutoAugment`, which
we use as a helper class to help us apply AutoAugment policies to datasets.

This is a tutorial (in the sense describe in https://documentation.divio.com/structure/).

For an example of how the material is used in our library, see the source code of
:meth:`AaLearner._test_autoaugment_policy <MetaAugment.autoaugment_learners.AaLearner>`.

Let's say we have a policy within the search space specified by the original 
AutoAugment paper:

.. code-block::

    my_policy = [
                        (("Invert", 0.8, None), ("Contrast", 0.2, 6)),
                        (("Rotate", 0.7, 2), ("Invert", 0.8, None)),
                        (("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)),
                        (("ShearY", 0.5, 8), ("Invert", 0.7, None)),
                        (("AutoContrast", 0.5, None), ("Equalize", 0.9, None))
                        ]

And that we also have a dataset that we want to apply this policy to:

.. code-block::

    train_dataset = datasets.MNIST(root='./datasets/mnist/train', train=True)
    test_dataset = datasets.MNIST(root='./datasets/mnist/test', train=False,
                                transform=torchvision.transforms.ToTensor())

The ``train_dataset`` object will have an attribute ``.transform`` with the 
default value ``None``.
The ``.transform`` attribute takes a function which takes an image as an input 
and returns a transformed image.

We need a function which will apply the ``my_policy`` and we use 
an ``AutoAugment`` for this job.

.. code-block::
    :caption: Creating an ``AutoAugment`` object and imbueing it with ``my_policy``.

    aa_transform = AutoAugment()
    aa_transform.subpolicies = my_policy
    train_transform = transforms.Compose([
                                            aa_transform,
                                            transforms.ToTensor()
                                        ])

We can use ``train_transform`` as an image function:

.. code-block::
    :caption: This function call will return an augmented image

    augmented_image = train_transform(original_image)

We usually apply an image function to a ``Dataset`` like this:

.. code-block::

    train_dataset = datasets.MNIST(root='./datasets/mnist/train', train=True, transform=my_function)

However, in our library we often have to apply a image function *after* the ``Dataset`` 
object was already created. (For example, a ``Dataset`` object is created and trained on
multiple times using different policies).
In this case, we alter the ``.transform`` attribute:

.. code-block::

    train_dataset.transform = train_transform

Now if we can create a ``DataLoader`` object from ``train_dataset``, it will automatically
apply ``my_policy``.