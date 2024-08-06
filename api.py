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
    model.fit(dataset, epochs=100, lr=2e-3, weight_decay=1e-4)
    return 'Model trained!', 200

@app.route('/diagnose', methods=['GET'])
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

@app.route('/recommend', methods=['GET'])
def recommend():
    global model
    if model is None:
        return 'Model not trained!', 400
    queries: list[dict] = request.json

    response = []
    for content in queries:
        not_valid_questions = data.get_answered_questions(content['studentID'])
        not_valid_questions = data.to_numerical_ids(not_valid_questions, encoder='question')
        
        student_id = data.to_numerical_ids(content['studentID'], encoder='user')[0]
        skill = data.to_numerical_ids(content['skill'], encoder='skill')[0]
        threshold = content['threshold']
        top_k = content.get('topK', 5)
        recommended_questions = model.recommend(student_id=student_id, skill=skill, threshold=threshold, top_k=top_k, not_valid_questions=not_valid_questions)
        recommended_questions['question_id'] = data.to_categorical_ids(recommended_questions['question_id'], encoder='question')
        response.append({
            'studentID': content['studentID'],
            'skill': content['skill'],
            'threshold': threshold,
            'recommendations': recommended_questions.to_dict(orient='records')
        })

    return jsonify(response), 200

if __name__ == '__main__':
    app.run(port=5000)