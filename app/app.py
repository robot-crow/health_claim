from flask import Flask, request, Blueprint
import logging
from pages.index import index_page
from pages.predict import predict_page

logging.basicConfig(filename='dump.log',level=logging.INFO)

app = Flask(__name__)
app.register_blueprint(index_page)
app.register_blueprint(predict_page)


if __name__ == "__main__":
    app.run(debug=True)
