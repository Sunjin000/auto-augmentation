from MetaAugment.autoaugment_learners.aa_learner import augmentation_space, aa_learner
from MetaAugment.autoaugment_learners.evo_learner import evo_learner
from MetaAugment.autoaugment_learners.gru_learner import gru_learner
from MetaAugment.autoaugment_learners.randomsearch_learner import randomsearch_learner

def gen_learner(name: str, **kwags):
    """
    Generates a learner based on input from the user. 

    Parameters
    -------------
    name -> string
        Of form 'gru', 'rand', 'evo' for GRU, Random search, or Evolutionary 
        learner generation

    **kwags -> key word arguments for respective learner
    """
    name == name.lower()
    if name == "gru":
        agent = gru_learner(**kwags)
    elif name == "evo":
        agent = evo_learner(**kwags)
    elif name == "rand" or name == "random" or name == "ran":
        agent = randomsearch_learner(**kwags)
    return agent
