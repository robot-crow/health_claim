from flask import Blueprint

index_page = Blueprint('index_page', __name__)

@index_page.route("/",methods=["GET"])
def index():
    return "<h1>Hello World!</h1>"
