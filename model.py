import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NeuralCD(nn.Module):
    def __init__(self, q_matrix: np.ndarray, num_students: int, layers: list = [512, 256], dropout: float = 0.5):
        super(NeuralCD, self).__init__()
        self.num_students = num_students

        self.Q = nn.Embedding.from_pretrained(
            torch.tensor(q_matrix, dtype=torch.float32), freeze=True)
        self.A = nn.Embedding(num_students, q_matrix.shape[1])
        self.B = nn.Embedding(q_matrix.shape[0], q_matrix.shape[1])
        self.D = nn.Embedding(q_matrix.shape[0], 1)

        self.interaction_function = nn.Sequential()
        for i, layer in enumerate(layers):
            self.interaction_function.add_module(f"linear_{i}", nn.Linear(layers[i-1] if i > 0 else q_matrix.shape[1], layer))
            self.interaction_function.add_module(f"activation_{i}", nn.Sigmoid())
            self.interaction_function.add_module(f"dropout_{i}", nn.Dropout(dropout))
        self.interaction_function.add_module(f"output", nn.Linear(layers[-1], 1))

        self.apply(self.__init_weights)

    def __init_weights(self, m):
        if isinstance(m, nn.Embedding) and m.weight.requires_grad:
            nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.Linear) and m.weight.requires_grad:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    @torch.no_grad()
    def __clip(self, x):
        if isinstance(x, nn.Linear):
            x.weight.clamp_(min=0)

    def forward(self, student_id: torch.Tensor, question_id: torch.Tensor):
        h_s: torch.Tensor = torch.sigmoid(self.A(student_id))
        Q_e: torch.Tensor = self.Q(question_id)
        h_diff: torch.Tensor = torch.sigmoid(self.B(question_id))
        h_disc: torch.Tensor = torch.sigmoid(self.D(question_id)) * 10

        x: torch.Tensor = Q_e * (h_s - h_diff) * h_disc
        x = self.interaction_function(x)
        return torch.sigmoid(x).view(-1)
    
    def get_user_embedding(self, student_id):
        i = torch.tensor(student_id, dtype=torch.long).to(DEVICE)
        return torch.sigmoid(self.A(i)).detach().cpu().numpy()
    
    def get_question_embedding(self, question_id):
        question_id = torch.tensor(question_id, dtype=torch.long).to(DEVICE)
        difficulty = torch.sigmoid(self.B(question_id)).detach().cpu().numpy()
        discrimination = torch.sigmoid(self.D(question_id)).detach().cpu().numpy()
        return difficulty, discrimination 

    def fit(self, dataset: Dataset, epochs: int = 10, batch_size: int = 32, lr: float = 0.001, weight_decay: float = 0.0):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCELoss()

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for _ in (bar := tqdm(range(epochs))):
            epoch_loss = 0
            for student_id, question_id, answer in dataloader:
                optimizer.zero_grad()
                student_id, question_id, answer = student_id.to(DEVICE), question_id.to(DEVICE), answer.to(DEVICE)
                y_pred = self(student_id, question_id)
                loss = criterion(y_pred, answer)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                # clip weights to [0, \infty) to guarantee monotonicity assumption
                self.interaction_function.apply(self.__clip)
            
            bar.set_postfix(loss=epoch_loss)
        
        self.eval()
    
    @torch.no_grad()
    def recommend(self, student_id: int, skill: int, top_k: int = 5, not_valid_questions: list = []):
        skill = torch.tensor(skill, dtype=torch.long).to(DEVICE)

        # get valid questions
        valid_questions = torch.tensor(
            np.setdiff1d(
                ar1=np.arange(self.Q.weight.shape[0]), 
                ar2=not_valid_questions, 
                assume_unique=False
            ), 
            dtype=torch.long
        ).to(DEVICE)

        # get all valid questions for the skill
        candidates = self.Q.weight[valid_questions, skill].nonzero().view(-1)

        # calculate the probability of answering correctly for each question
        student_ids = torch.tensor(len(candidates) * [student_id], dtype=torch.long).to(DEVICE)
        probs = self(student_ids, candidates).detach().cpu().numpy()

        # get threshold for the skill from the user embedding
        threshold = 1 - self.get_user_embedding(student_id)[skill]

        # calculate the difference between the threshold and the probabilities
        diff = np.abs(probs - threshold)

        #get top k questions
        return candidates[np.argsort(diff)][:top_k].detach().cpu().numpy()

class CDataset(Dataset):
    def __init__(self, logs: np.ndarray):
        self.logs = logs

    def __len__(self):
        return len(self.logs)

    def __getitem__(self, idx):
        x = torch.tensor(self.logs[idx, :], dtype=torch.long)
        return x[0], x[1], x[2].float()