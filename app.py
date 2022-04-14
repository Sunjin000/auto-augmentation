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


UPLOAD_FOLDER = '/datasets'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'pdf', 'py'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/user_input', methods = ['GET', 'POST'])
def upload_file():
    print("HELLoasdjsadojsadojsaodjsaoij")
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file', filename=filename))
    return '''
    
    '''


if __name__ == '__main__':
    app.run(debug=True)