from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from math import radians, cos, sin, sqrt, atan2
import json

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------- Helpers ----------------------
def load_data(file_path):
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".json"):
        with open(file_path) as f:
            return pd.json_normalize(json.load(f))
    else:
        raise ValueError("Unsupported file format")

def haversine(coord1, coord2):
    R = 6371000
    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def compute_matrix(coords):
    n = len(coords)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i][j] = haversine(coords[i], coords[j])
    return matrix

# ---------------------- API Route ----------------------
@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        data = load_data(file_path)

        if "consignee.latitude" not in data.columns or "consignee.longitude" not in data.columns:
            return jsonify({"error": "Missing latitude or longitude columns"}), 400

        coords_df = data[["consignee.latitude", "consignee.longitude"]].drop_duplicates().reset_index(drop=True)
        coords = list(coords_df.itertuples(index=False, name=None))
        starting_point = (19.0823107, 73.1452029)
        locations = [starting_point] + coords

        dist_matrix_full = compute_matrix(locations)
        dist_matrix_coords = compute_matrix(coords)

        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            linkage='complete',
            distance_threshold=10000
        )
        cluster_labels = clustering.fit_predict(dist_matrix_coords)
        clusters = np.insert(cluster_labels, 0, 0)

        alpha = 1000
        penalized_matrix = [
            [int(dist_matrix_full[i][j] + (alpha if clusters[i] != clusters[j] else 0))
             for j in range(len(locations))]
            for i in range(len(locations))
        ]

        # Routing
        manager = pywrapcp.RoutingIndexManager(len(locations), 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            return penalized_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.time_limit.seconds = 10

        solution = routing.SolveWithParameters(search_parameters)

        if not solution:
            return jsonify({"error": "No route found"}), 500

        # Get optimized route
        index = routing.Start(0)
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))

        optimized_path = [locations[i] for i in route]
        reversed_path = [optimized_path[0]] + optimized_path[1:][::-1]

        optimized_dist = sum(haversine(optimized_path[i], optimized_path[i + 1]) for i in range(len(optimized_path) - 1))
        reversed_dist = sum(haversine(reversed_path[i], reversed_path[i + 1]) for i in range(len(reversed_path) - 1))

        if optimized_dist <= reversed_dist:
            best_path = optimized_path
            label = "optimized"
        else:
            best_path = reversed_path
            label = "reversed"

        return jsonify({
            "path": [[lat, lon] for lat, lon in best_path],
            "distance_km": round(min(optimized_dist, reversed_dist) / 1000, 2),
            "type": label
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    


# -------------------------------------------------------------------------

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///register.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False  # ✅ Fixed typo
db = SQLAlchemy(app)

class Register(db.Model):  # ✅ Capitalized class name (Python convention)
    email = db.Column(db.String(120), primary_key=True)
    password = db.Column(db.String(120), nullable=False)
    username = db.Column(db.String(120), nullable=False)

    def __repr__(self):
        return f"<Register {self.email}>"

@app.route("/register", methods=["POST"])
def Register_users():
    data = request.get_json()
    if not data:
        return jsonify({"message": "No input provided"}), 400

    email = data.get("email")
    password = data.get("password")
    username = data.get("username")

    if not all([email, password, username]):
        return jsonify({"message": "All fields are required"}), 400

    if Register.query.get(email):
        return jsonify({"message": "Email already registered"}), 409

    hashed_password = generate_password_hash(password)
    new_user = Register(email=email, username=username, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"message": "User registered successfully"}), 201


# ---------------------- Run Server ----------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # ✅ Must be before app.run()
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True,port=port)