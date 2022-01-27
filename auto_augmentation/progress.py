from flask import Blueprint, request, render_template, flash, send_file
import subprocess

bp = Blueprint("progress", __name__)

@bp.route("/user_input", methods=["GET", "POST"])
def response():
    
    return render_template("progress.html")