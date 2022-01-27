import os

from flask import Flask, render_template, request, flash

from auto_augmentation import home, progress, result, download_file

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    os.makedirs(app.instance_path, exist_ok=True)

    app.register_blueprint(home.bp)
    app.register_blueprint(progress.bp)
    app.register_blueprint(result.bp)
    app.register_blueprint(download_file.bp)
    

    return app
