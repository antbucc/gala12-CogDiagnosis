from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)
MODEL = None


def get_demo_data():
    SKILLS = ['Plastic', 'Bees', 'CFC', 'Detergents']
    QUESTIONS_PER_SKILL = 25

    Q_MATRIX = []
    for i in range(len(SKILLS)):
        rows = np.zeros((QUESTIONS_PER_SKILL, len(SKILLS)), dtype=int)
        rows[:, i] = 1
        Q_MATRIX.append(rows)
    Q_MATRIX = np.vstack(Q_MATRIX)  # Skill x Question matrix
    DIFFICULTY = np.random.uniform(0, 1, Q_MATRIX.shape[0])
    STUDENTS = 20
    QUESTIONS_PER_USER = 20

    logs = []
    for i in range(STUDENTS):
        questions = np.random.choice(
            Q_MATRIX.shape[0], QUESTIONS_PER_USER, replace=False)
        answers = np.random.binomial(1, DIFFICULTY[questions])
        logs.append([(i, questions[j], answers[j])
                    for j in range(QUESTIONS_PER_USER)])

    logs = np.vstack(logs)
    return logs, Q_MATRIX


@app.route('/train', methods=['GET'])
def train():
    global MODEL
    logs, Q_MATRIX = get_demo_data()
    dataset = CDataset(logs)
    users = len(np.unique(logs[:, 0]))
    MODEL = NeuralCD(Q_MATRIX, num_students=users)
    MODEL = MODEL.to(DEVICE)
    MODEL.fit(dataset)
    return 'Model trained successfully', 200


@app.route('/user_embedding', methods=['POST'])
def user_embedding():
    global MODEL
    if MODEL is None:
        return 'Model not trained', 400
    data = request.json
    student_id = data['student_id']
    embs = MODEL.get_user_embedding(student_id)
    return jsonify([
        {'student_id': i, 'skills': emb}
        for i, emb in zip(student_id, embs.tolist())
    ]), 200


@app.route('/recommend', methods=['POST'])
def recommend():
    global MODEL
    if MODEL is None:
        return 'Model not trained', 400
    data = request.json

if __name__ == '__main__':
    app.run(port=5000)
