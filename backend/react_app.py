from flask import Flask, request
# from flask_cors import CORS

app = Flask(__name__)


@app.route('/home', methods=["GET", "POST"])
def home():
    print('in flask home')
    data = request.get('')

    return data
