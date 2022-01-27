from flask import Blueprint, request, render_template, flash, send_file
import subprocess

bp = Blueprint("search_result", __name__)

@bp.route("/search_result", methods=["GET", "POST"])
@bp.route("/choose_type", methods=["GET", "POST"])
def response():
    query = request.args["search_query"]
    if not query:
        flash("No query")
    print('query', query)

    return render_template("result.html")