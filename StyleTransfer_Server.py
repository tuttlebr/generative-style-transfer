import StyleTransfer
from flask import Flask, request
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)


class DoTransfer(Resource):
    def get(self, content_image, style_image):
        return {"data": StyleTransfer.st_model(content_image, style_image)}


api.add_resource(DoTransfer, "/generative/<content_image>/<style_image>")

if __name__ == "__main__":
    app.run()
