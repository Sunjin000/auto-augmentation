from flask import Blueprint, request, render_template, flash, send_file

bp = Blueprint("download_file", __name__)

@bp.route("/download_file", methods=["GET"])
@bp.route("/download", methods=["GET", "POST"])
def download():    
    # Setup for the 'return send_file()' function call at the end of this function
    path = 'templates/CNN.zip' # e.g. 'templates/download.markdown'

    return send_file(path,
                    as_attachment=True)
