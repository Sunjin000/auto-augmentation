from flask import Flask, request
# from flask_cors import CORS

app = Flask(__name__)


@app.route('/profile')
def my_profile():
    response_body = {
        "name": "Nagato",
        "about" :"Hello! I'm a full stack developer that loves python and javascript"
    }

    return response_body

# def get_user_input():

#     return request.args