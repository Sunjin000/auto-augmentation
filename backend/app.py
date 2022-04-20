
# if __name__ == '__main__':
#     app.run(host='0.0.0.0',port=port)

from auto_augmentation import home, progress,result, training


from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from auto_augmentation import create_app
import os

app = create_app()




# app = Flask(__name__)
# app.config.from_mapping(
#     SECRET_KEY='dev',
# )
# # ensure the instance folder exists
# os.makedirs(app.instance_path, exist_ok=True)
# from auto_augmentation import download_file
# app.register_blueprint(home.bp)
# app.register_blueprint(progress.bp)
# app.register_blueprint(training.bp)
# app.register_blueprint(result.bp)
# app.register_blueprint(download_file.bp)



# port = int(os.environ.get("PORT", 5000))




if __name__ == '__main__':
    app.run(debug=False)