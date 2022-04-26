from dataclasses import dataclass
from flask import Flask, request, current_app, send_file, send_from_directory, redirect, url_for, session
from flask_cors import CORS, cross_origin

import os
import zipfile

import torch
from numpy import save, load
# import temp_util.wapp_util as wapp_util
import time

import os
import sys
sys.path.insert(0, os.path.abspath('..'))
torch.manual_seed(0)

print('@@@ import successful')

app = Flask(__name__, static_folder='react_frontend/build', static_url_path='/')
CORS(app)

# it is used to collect user input and store them in the app
@app.route('/home', methods=["GET", "POST"])
# @cross_origin()
def get_form_data():
    
    if request.method == 'POST':
        print('@@@ in Flask Home')
        
        form_data = request.form
        print('@@@ this is form data', form_data)

        # required input
        ds = form_data['select_dataset'] # pick dataset (MNIST, KMNIST, FashionMNIST, CIFAR10, CIFAR100)
        IsLeNet = form_data["select_network"]   # using LeNet or EasyNet or SimpleNet ->> default 
        auto_aug_learner = form_data["select_learner"] # augmentation methods to be excluded

        print('@@@ required user input:', 'ds', ds, 'IsLeNet:', IsLeNet, 'auto_aug_leanrer:',auto_aug_learner)
        # advanced input
        if form_data['batch_size'] != 'undefined': 
            batch_size = form_data['batch_size']       # size of batch the inner NN is trained with
        else: 
            batch_size = 1 # this is for demonstration purposes
        if form_data['learning_rate'] != 'undefined': 
            learning_rate =  form_data['learning_rate']  # fix learning rate
        else: 
            learning_rate = 10-1
        if form_data['toy_size'] != 'undefined': 
            toy_size = form_data['toy_size']      # total propeortion of training and test set we use
        else: 
            toy_size = 1 # this is for demonstration purposes
        if form_data['iterations'] != 'undefined': 
            iterations = form_data['iterations']      # total iterations, should be more than the number of policies
        else: 
            iterations = 10
        exclude_method = form_data['select_action']
        print('@@@ advanced search: batch_size:', batch_size, 'learning_rate:', learning_rate, 'toy_size:', toy_size, 'iterations:', iterations, 'exclude_method', exclude_method)
        

        # default values 
        max_epochs = 10      # max number of epochs that is run if early stopping is not hit
        early_stop_num = 10   # max number of worse validation scores before early stopping is triggered
        num_policies = 5      # fix number of policies
        num_sub_policies = 5  # fix number of sub-policies in a policy
        
        
        # if user upload datasets and networks, save them in the database
        if ds == 'Other':
            ds_folder = request.files['ds_upload'] 
            print('!!!ds_folder', ds_folder)
            ds_name_zip = ds_folder.filename
            ds_name = ds_name_zip.split('.')[0]
            ds_folder.save('./datasets/'+ ds_name_zip)
            with zipfile.ZipFile('./datasets/'+ ds_name_zip, 'r') as zip_ref:
                zip_ref.extractall('./datasets/upload_dataset/')
            if not current_app.debug:
                os.remove(f'./datasets/{ds_name_zip}')
        else: 
            ds_name_zip = None
            ds_name = None

        # test if uploaded dataset meets the criteria 
        for (dirpath, dirnames, filenames) in os.walk(f'./datasets/upload_dataset/{ds_name}/'):
            for dirname in dirnames:
                if dirname[0:6] != 'class_':
                    return None # neet to change render to a 'failed dataset webpage'

        # save the user uploaded network
        if IsLeNet == 'Other':
            childnetwork = request.files['network_upload']
            childnetwork.save('./child_networks/'+childnetwork.filename)
            network_name = childnetwork.filename
        else: 
            network_name = None

        
        print("@@@ user input has all stored in the app")

        data = {'ds': ds, 'ds_name': ds_name_zip, 'IsLeNet': IsLeNet, 'network_name': network_name,
                'auto_aug_learner':auto_aug_learner, 'batch_size': batch_size, 'learning_rate': learning_rate, 
                'toy_size':toy_size, 'iterations':iterations, 'exclude_method': exclude_method, }

        current_app.config['data'] = data
        
        print('@@@ all data sent', current_app.config['data'])

    # try this if you want it might work, it might not
    # wapp_util.parse_users_learner_spec(
    #                         num_policies,
    #                         num_sub_policies,
    #                         early_stop_num,
    #                         max_epochs,
    #                         **data,
    #                         )
    else: 
        print('it is GET method')
    data = current_app.config['data']
    return data
    # return redirect(url_for('confirm', data=data))




# ========================================================================
@app.route('/confirm', methods=['POST', 'GET'])
@cross_origin()
def confirm():  
    print('inside confirm page')
    data = current_app.config['data']
    print("current_app.config['data']", current_app.config['data'])
    # print("session.get('data')", session.get('data'))
    # data = request.args.get('data')
    return data




# ========================================================================
@app.route('/training', methods=['POST', 'GET'])
@cross_origin()
def training():

    # default values 
    max_epochs = 10      # max number of epochs that is run if early stopping is not hit
    early_stop_num = 10   # max number of worse validation scores before early stopping is triggered
    num_policies = 5      # fix number of policies
    num_sub_policies = 5  # fix number of sub-policies in a policy
    data = current_app.config.get('data')

    # fake training
    print('pretend it is training')
    time.sleep(1)
    print('epoch: 1')
    time.sleep(1)
    print('epoch: 2')
    time.sleep(1) 
    print('epoch: 3')
    print('it has finished training')

    return {'status': 'Training is done!'}


# ========================================================================
@app.route('/result')
@cross_origin()
def show_result():
    file_path = "./react_backend/policy.txt"
    f = open(file_path, "r")
    return send_file(file_path, as_attachment=True)



@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')



if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)