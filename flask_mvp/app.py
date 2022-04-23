# from flask import Flask
# from auto_augmentation import create_app
# import os

# app = create_app()
# port = int(os.environ.get("PORT", 5000))

# if __name__ == '__main__':
#     app.run(host='0.0.0.0',port=port)


from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from auto_augmentation import create_app
import os
app = create_app()
port = int(os.environ.get("PORT", 5000))


if __name__ == '__main__':
    app.run(debug=True)