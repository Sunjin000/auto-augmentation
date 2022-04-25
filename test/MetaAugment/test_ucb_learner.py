import MetaAugment.autoaugment_learners as aal


def test_ucb_learner():
    learner = aal.ucb_learner(
        # parameters that define the search space
                sp_num=5,
                p_bins=11,
                m_bins=10,
                discrete_p_m=True,
                # hyperparameters for when training the child_network
                batch_size=8,
                toy_flag=False,
                toy_size=0.1,
                learning_rate=1e-1,
                max_epochs=float('inf'),
                early_stop_num=30,
                # ucb_learner specific hyperparameter
                num_policies=100
    )
    print(learner.policies)

if __name__=="__main__":
    test_ucb_learner()