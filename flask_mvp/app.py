
# if __name__ == '__main__':
#     app.run(host='0.0.0.0',port=port)

from numpy import broadcast
from auto_augmentation import home, progress,result
from flask_mvp.auto_augmentation import training
from flask_socketio import SocketIO,  send

from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from auto_augmentation import create_app
import os

app = create_app()


socketio = SocketIO(app)



@socketio.on('message')
def handleMessage(msg):
    print("Message: ", msg)
    send(msg, broadcast=True)



if __name__ == '__main__':
    app.run(debug=True)
    # socketio.run(app)