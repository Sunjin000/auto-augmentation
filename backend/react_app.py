from flask import Flask, request
# from flask_cors import CORS

app = Flask(__name__)


@app.route('/home', methods=["POST"])
def home():
    data = request.json

    return data
