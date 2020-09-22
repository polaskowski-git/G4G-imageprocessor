#pylint: disable=missing-module-docstring
#pylint: disable=missing-class-docstring
#pylint: disable=missing-function-docstring
#pylint: disable-msg=too-many-arguments
#pylint: disable-msg=no-self-use

from flask import Flask
from flask_restful import Api, Resource, reqparse, abort
import g4g

def abort_if_id_exists(training_id):
    if training_id in trainings:
        abort(409, message="Identyfikator już istnieje")

app = Flask(__name__)
api = Api(app)

training_post_args = reqparse.RequestParser()
training_post_args.add_argument("training_id", type=int,
                                help="Numer identyfikacyjny treningu (liczba)", required=True)
training_post_args.add_argument("image", type=str,
                                help="Zdjęcie tarczy w formacie Base64 (tekst)", required=True)
training_post_args.add_argument("caliber", type=int,
                                help="Kaliber podany w mm (liczba)", required=True)
training_post_args.add_argument("magazine_capacity", type=int,
                                help="Wielkość magazynku (liczba)", required=True)
training_post_args.add_argument("distance_to_target", type=int,
                                help="Dystans od tarczy w metrach (liczba)", required=True)

trainings = {}

class Training(Resource):
    def post(self, training_id):
        abort_if_id_exists(training_id)
        args = training_post_args.parse_args()
        trainings[training_id] = args
        #print(args['caliber'])
        results = g4g.process_image(args['image'], args['caliber'],
                                    args['magazine_capacity'], args['distance_to_target'])
        return results, 201

api.add_resource(Training, "/training/<int:training_id>")

if __name__ == "__main__":
    app.run(debug=True)
