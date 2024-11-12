import os
import csv
import numpy as np
from sklearn.utils import shuffle
import ray

def pad(data, target_length, target_value=0):
    return np.pad(data, (0, target_length - len(data)), 'constant', constant_values=target_value)

def one_hot(indices, depth):
    encoding = np.concatenate((np.eye(depth), [np.zeros(depth)]))
    indices[indices!=-1] -= 1
    return encoding[indices]


class OriginalInputProcessor(object):
    def process_problems_and_corrects(self, problem_seqs, correct_seqs, num_problems, is_train=True):
        """
        This function aims to process the problem sequence and the correct sequence into a DKT feedable X and y.
        :param problem_seqs: it is in shape [batch_size, None]
        :param correct_seqs: it is the same shape as problem_seqs
        :return:
        """
        # pad the sequence with the maximum sequence length
        max_seq_length = max([len(problem) for problem in problem_seqs])
        problem_seqs_pad = np.array([pad(problem, max_seq_length, target_value=-1) for problem in problem_seqs])
        correct_seqs_pad = np.array([pad(correct, max_seq_length, target_value=-1) for correct in correct_seqs])

        # find the correct seqs matrix as the following way:
        # Let problem_seq = [1,3,2,-1,-1] as a and correct_seq = [1,0,1,-1,-1] as b, which are padded already
        # First, find the element-wise multiplication of a*b*b = [1,0,2,-1,-1]
        # Then, for any values 0, assign it to -1 in the vector = [1,-1,2,-1,-1] as c
        # Such that when we one hot encoding the vector c, it will results a zero vector
        temp = problem_seqs_pad * correct_seqs_pad * correct_seqs_pad  # temp is c in the comment.
        temp[temp == 0] = -1
        correct_seqs_pad = temp

        # one hot encode the information
        problem_seqs_oh = one_hot(problem_seqs_pad, depth=num_problems)
        correct_seqs_oh = one_hot(correct_seqs_pad, depth=num_problems)

        # slice out the x and y
        if is_train:
            x_problem_seqs = problem_seqs_oh[:, :-1]
            x_correct_seqs = correct_seqs_oh[:, :-1]
            y_problem_seqs = problem_seqs_oh[:, 1:]
            y_correct_seqs = correct_seqs_oh[:, 1:]
        else:
            x_problem_seqs = problem_seqs_oh[:, :]
            x_correct_seqs = correct_seqs_oh[:, :]
            y_problem_seqs = problem_seqs_oh[:, :]
            y_correct_seqs = correct_seqs_oh[:, :]

        X = np.concatenate((x_problem_seqs, x_correct_seqs), axis=2)

        result = (X, y_problem_seqs, y_correct_seqs)
        return result

@ray.remote
class BatchGenerator:
    """
    Generate batch for DKT model
    """

    def __init__(self, problem_seqs, correct_seqs, num_problems, batch_size, input_processor=OriginalInputProcessor(),
                 **kwargs):
        self.cursor = 0  # point to the current batch index
        self.problem_seqs = problem_seqs
        self.correct_seqs = correct_seqs
        self.batch_size = batch_size
        self.num_problems = num_problems
        self.num_samples = len(problem_seqs)
        self.num_batches = len(problem_seqs) // batch_size + 1 if len(problem_seqs) % batch_size != 0 else len(problem_seqs) // batch_size
        self.input_processor = input_processor
        self._current_batch = None

    def next_batch(self, is_train=True):
        start_idx = self.cursor * self.batch_size
        end_idx = min((self.cursor + 1) * self.batch_size, self.num_samples)
        problem_seqs = self.problem_seqs[start_idx:end_idx]
        correct_seqs = self.correct_seqs[start_idx:end_idx]

        # x_problem_seqs, x_correct_seqs, y_problem_seqs, y_correct_seqs
        self._current_batch = self.input_processor.process_problems_and_corrects(problem_seqs,
                                                                                 correct_seqs,
                                                                                 self.num_problems,
                                                                                 is_train=is_train)
        self._update_cursor()
        return self._current_batch

    def get_num_batches(self):
        """
        Return the number of batches based on the data size and batch size.
        """
        return self.num_batches

    @property
    def current_batch(self):
        if self._current_batch is None:
            print("Current batch is None.")
        return None

    def _update_cursor(self):
        self.cursor = (self.cursor + 1) % self.num_batches

    def reset_cursor(self):
        self.cursor = 0

    def shuffle(self):
        self.problem_seqs, self.correct_seqs = shuffle(self.problem_seqs, self.correct_seqs, random_state=42)



def read_data_from_csv(filename):
    # read the csv file
    rows = []
    # Open the file with the correct encoding
    with open(filename, 'r', encoding='utf-8') as f:
        print("Reading {0}".format(filename))
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            rows.append(row)
        print("{0} lines was read".format(len(rows)))


    max_seq_length = 0
    num_problems = 5002
    
    tuples = []
    for i in range(0, len(rows), 3):
        seq_length = len(rows[i + 1])

        # only keep student with at least 3 records.
        if seq_length < 3:
            continue

        problem_seq = rows[i + 1]
        correct_seq = rows[i + 2]

        # Identify the invalid problem IDs
        invalid_ids_loc = [i for i, pid in enumerate(problem_seq) if pid == '']

        # Delete from the end to avoid index issues
        for invalid_loc in sorted(invalid_ids_loc, reverse=True):
            if invalid_loc < len(problem_seq):
                del problem_seq[invalid_loc]
                del correct_seq[invalid_loc]

        # Convert problem_seq from string to int, handle floating point values
        try:
            problem_seq = [int(float(pid)) for pid in problem_seq]
        except ValueError as e:
            print(f"Error converting problem_seq: {problem_seq}, Error: {e}")
            continue  # Skip this sequence if there's an issue

        # Convert correct_seq from string to int, handle floating point values
        try:
            correct_seq = [int(float(c)) for c in correct_seq]
        except ValueError as e:
            print(f"Error converting correct_seq: {correct_seq}, Error: {e}")
            continue  # Skip this sequence if there's an issue

        tup = (seq_length, problem_seq, correct_seq)
        tuples.append(tup)

        if max_seq_length < seq_length:
            max_seq_length = seq_length

    print("max_num_problems_answered:", max_seq_length)
    print("num_problems:", num_problems)
    print("The number of data is {0}".format(len(tuples)))
    print("Finish reading data.")

    return tuples, num_problems, max_seq_length


class DKTData:
    def __init__(self, train_path, valid_path, test_path, batch_size=32):
        self.students_train, num_problems_train, max_seq_length_train = read_data_from_csv(train_path)
        self.students_valid, num_problems_valid, max_seq_length_valid = read_data_from_csv(valid_path)
        self.students_test, num_problems_test, max_seq_length_test = read_data_from_csv(test_path)
        self.num_problems = max(num_problems_test, num_problems_train, num_problems_valid)
        self.max_seq_length = max(max_seq_length_train, max_seq_length_test, max_seq_length_valid)

        problem_seqs = [student[1] for student in self.students_train]
        correct_seqs = [student[2] for student in self.students_train]
        self.train = BatchGenerator.remote(problem_seqs, correct_seqs, self.num_problems, batch_size)  # remote() 사용

        problem_seqs = [student[1] for student in self.students_valid]
        correct_seqs = [student[2] for student in self.students_valid]
        self.valid = BatchGenerator.remote(problem_seqs, correct_seqs, self.num_problems, batch_size)  # remote() 사용
        
        problem_seqs = [student[1] for student in self.students_test]
        correct_seqs = [student[2] for student in self.students_test]
        self.test = BatchGenerator.remote(problem_seqs, correct_seqs, self.num_problems, batch_size)  # remote() 사용
