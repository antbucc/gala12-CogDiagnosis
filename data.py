import os

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import OrdinalEncoder

_BASE_URL = os.environ.get(
    'API_ENDPOINT', 'https://gala24demo-api-production.up.railway.app')


def _get_data(endpoint):
    url = f"{_BASE_URL}/{endpoint}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    return data


class LearningData:
    def __init__(self):
        self.__user_encoder = OrdinalEncoder()
        self.__question_encoder = OrdinalEncoder()
        self.__skill_encoder = OrdinalEncoder()

    def initiate(self):
        self.__fetch_q_matrix()
        self.__fetch_response_logs()
        self.__collect_answered_questions()

    def get_n_users(self):
        return len(self.__user_encoder.categories_[0])

    def get_skills(self):
        return self.__skill_encoder.categories_[0]

    def as_arrays(self):
        logs = self.__response_logs.copy()
        logs['studentID'] = self.__user_encoder.transform(
            logs['studentID'].values.reshape(-1, 1))
        logs['questionID'] = self.__question_encoder.transform(
            logs['questionID'].values.reshape(-1, 1))

        return self.__q_matrix.values, logs.values

    def get_answered_questions(self, student_id):
        return self.__answered_questions[student_id]

    def to_numerical_ids(self, ids, encoder):
        encoder = {
            'user': self.__user_encoder,
            'question': self.__question_encoder,
            'skill': self.__skill_encoder
        }[encoder]

        return encoder.transform(np.array(ids).reshape(-1, 1)).flatten()
    
    def to_categorical_ids(self, ids, encoder):
        encoder = {
            'user': self.__user_encoder,
            'question': self.__question_encoder,
            'skill': self.__skill_encoder
        }[encoder]

        return encoder.inverse_transform(np.array(ids).reshape(-1, 1)).flatten()

    def __fetch_q_matrix(self):
        questions = _get_data('questions')['activities']
        questions = pd.Series({q['id']: q['skillIDs'] for q in questions})
        all_skills = [s['_id'] for s in _get_data('skills')['skills']]

        self.__q_matrix = pd.DataFrame(
            index=questions.index, columns=all_skills, data=0)
        for q_id, skills in questions.items():
            self.__q_matrix.loc[q_id, skills] = 1

        self.__question_encoder.fit(
            self.__q_matrix.index.values.reshape(-1, 1))
        
        self.__skill_encoder.fit(
            self.__q_matrix.columns.values.reshape(-1, 1))

    def __fetch_response_logs(self):
        logs = _get_data('students-logs')
        logs = pd.DataFrame(logs)

        self.__user_encoder.fit(logs['studentID'].values.reshape(-1, 1))
        logs['response'] = logs['response'].apply(
            lambda x: 1 if x == 'True' else 0).astype(int)

        self.__response_logs = logs

    def __collect_answered_questions(self):
        self.__answered_questions: dict[str, np.ndarray] = {
            student_id: np.unique(
                self.__response_logs.loc[
                    (self.__response_logs['studentID'] == student_id)
                    & (self.__response_logs['response'] == 1), 'questionID'].values
            )
            for student_id in self.__user_encoder.categories_[0]
        }
