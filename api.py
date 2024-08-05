from flask import Flask, request, jsonify
from data import LearningData
from model import NeuralCD, CDataset, DEVICE

app = Flask(__name__)
data = LearningData()
model = None

@app.route('/train', methods=['GET'])
def train():
    global model
    data.initiate()
    q_matrix, logs = data.as_arrays()
    dataset = CDataset(logs)
    model = NeuralCD(q_matrix, data.get_n_users()).to(DEVICE)
    model.fit(dataset)
    return 'Model trained!', 200

@app.route('/diagnose', methods=['POST'])
def diagnose():
    global model
    if model is None:
        return 'Model not trained!', 400
    content = request.json
    ids = [content['studentID']] if type(content['studentID']) == str else content['studentID']
    student_ids = data.to_numerical_ids(ids, encoder='user')

    embeddings = model.get_user_embedding(student_ids)
    skills = data.get_skills()

    response = [
        {
            'studentID': student_id,
            'skills': {
                skill: value for skill, value in zip(skills, skill_values)
            }
        }
        for student_id, skill_values in zip(ids, embeddings.tolist())
    ]

    return jsonify(response), 200

@app.route('/recommend', methods=['POST'])
def recommend():
    global model
    if model is None:
        return 'Model not trained!', 400
    content: dict = request.json

    not_valid_questions = data.get_answered_questions(content['studentID'])
    student_id = data.to_numerical_ids(content['studentID'], encoder='user')[0]
    skill = data.to_numerical_ids(content['skill'], encoder='skill')[0]
    top_k = content.get('topK', 5)
    recommended_questions = model.recommend(student_id, skill, top_k, not_valid_questions)
    recommended_questions = data.to_categorical_ids(recommended_questions, encoder='question')

    return jsonify(recommended_questions.tolist()), 200

if __name__ == '__main__':
    app.run(port=5000, debug=True)