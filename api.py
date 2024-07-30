import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)
MODEL = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NeuralCD(nn.Module):
    def __init__(self, q_matrix: np.ndarray, num_students: int, num_layers: int = 3, hidden_size: int = 100):
        super(NeuralCD, self).__init__()
        self.num_students = num_students

        self.Q = nn.Embedding.from_pretrained(
            torch.tensor(q_matrix, dtype=torch.float32), freeze=True)
        self.A = nn.Embedding(num_students, q_matrix.shape[1])
        self.B = nn.Embedding(q_matrix.shape[0], q_matrix.shape[1])
        self.D = nn.Embedding(q_matrix.shape[0], 1)

        activation = nn.Sigmoid()
        self.interaction_function = nn.Sequential()
        self.interaction_function.add_module(
            'linear1', nn.Linear(q_matrix.shape[1], hidden_size))
        self.interaction_function.add_module('activation1', activation)

        for i in range(num_layers - 1):
            self.interaction_function.add_module(
                f'linear{i+2}', nn.Linear(hidden_size, hidden_size))
            self.interaction_function.add_module(
                f'activation{i+2}', activation)
        self.interaction_function.add_module(
            'linear_final', nn.Linear(hidden_size, 1))
        self.interaction_function.add_module('sigmoid', nn.Sigmoid())

    def forward(self, student_id: torch.Tensor, question_id: torch.Tensor):
        h_s: torch.Tensor = torch.sigmoid(self.A(student_id))
        Q_e: torch.Tensor = self.Q(question_id)
        h_diff: torch.Tensor = torch.sigmoid(self.B(question_id))
        h_disc: torch.Tensor = torch.sigmoid(self.D(question_id))

        x = Q_e * (h_s - h_diff) * h_disc
        x = self.interaction_function(x)
        return x.view(-1)

    def get_user_embedding(self, student_id):
        i = torch.tensor(student_id, dtype=torch.long).to(DEVICE)
        return torch.sigmoid(self.A(i)).detach().cpu().numpy()

    def predict(self, student_id: torch.Tensor, question_id: torch.Tensor):
        self.eval()
        with torch.no_grad():
            return self(student_id, question_id).detach().cpu().numpy()

    def fit(self, dataset: Dataset, epochs: int = 10, batch_size: int = 32, lr: float = 0.001, weight_decay: float = 0.0):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr,
                               weight_decay=weight_decay)
        criterion = nn.BCELoss()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for _ in (bar := tqdm(range(epochs))):
            for student_id, question_id, answer in dataloader:
                optimizer.zero_grad()
                student_id, question_id, answer = student_id.to(
                    DEVICE), question_id.to(DEVICE), answer.to(DEVICE)
                y_pred = self(student_id, question_id)
                loss = criterion(y_pred, answer)
                loss.backward()
                optimizer.step()
            bar.set_postfix(loss=loss.item())

        return self


class CDataset(Dataset):
    def __init__(self, logs: np.ndarray):
        self.logs = logs

    def __len__(self):
        return len(self.logs)

    def __getitem__(self, idx):
        x = torch.tensor(self.logs[idx, :], dtype=torch.long)
        return x[0], x[1], x[2].float()


def get_demo_data():
    np.random.seed(0)

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


@app.route('/predict', methods=['POST'])
def predict():
    global MODEL
    if MODEL is None:
        return 'Model not trained', 400
    data = request.json
    student_id = data['student_id']
    question_id = data['question_id']
    return jsonify({'prediction': MODEL.predict(torch.tensor(student_id), torch.tensor(question_id))}), 200


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


if __name__ == '__main__':
    app.run(port=5000, debug=True)
