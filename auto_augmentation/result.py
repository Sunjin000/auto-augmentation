from flask import Blueprint, request, render_template, flash, send_file
import subprocess

bp = Blueprint("result", __name__)

@bp.route("/show_result", methods=["GET", "POST"])
def response():
    
    return render_template("result.html")