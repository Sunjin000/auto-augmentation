from flask import Blueprint, request, render_template, flash, send_file, current_app
import os

import torch
torch.manual_seed(0)


import wapp_util

bp = Blueprint("training", __name__)


@bp.route("/start_training", methods=["GET", "POST"])
def response():



    # aa learner
    auto_aug_learner = current_app.config.get('AAL')
    # auto_aug_learner = session

    # search space & problem setting
    ds = current_app.config.get('ds')
    ds_name = current_app.config.get('DSN')
    exclude_method = current_app.config.get('exc_meth')
    num_funcs = current_app.config.get('NUMFUN')
    num_policies = current_app.config.get('NP')
    num_sub_policies = current_app.config.get('NSP')
    toy_size = current_app.config.get('TS')

    # child network
    IsLeNet = current_app.config.get('ISLENET')

    # child network training hyperparameters
    batch_size = current_app.config.get('BS')
    early_stop_num = current_app.config.get('ESN')
    iterations = current_app.config.get('IT')
    learning_rate = current_app.config.get('LR')
    max_epochs = current_app.config.get('ME')


    wapp_util.parse_users_learner_spec(
            auto_aug_learner, 
            ds, 
            ds_name, 
            exclude_method, 
            num_funcs, 
            num_policies, 
            num_sub_policies, 
            toy_size, 
            IsLeNet, 
            batch_size, 
            early_stop_num, 
            iterations, 
            learning_rate, 
            max_epochs
            )

    return render_template("progress.html", auto_aug_learner=auto_aug_learner)




