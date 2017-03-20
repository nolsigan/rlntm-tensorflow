import numpy as np


class Tape:

    def __init__(self, batch_size, tape_length, tape_size, table, initial=None):

        self.tape = np.zeros((batch_size, tape_length, tape_size), dtype=np.float32)
        self.ptr = np.zeros(batch_size, dtype=np.int32)
        self.table = table

        if initial is not None:
            self.tape = initial

    def read_tape(self):
        # ptr should always have values between 0 ~ self.tape.shape[1]-1

        result = np.zeros((self.tape.shape[0], self.tape.shape[2]))
        for batch in range(result.shape[0]):
            result[batch] = self.tape[batch][self.ptr[batch]]

        return result

    def write_tape(self, values, moves=None):

        for batch in range(self.tape.shape[0]):
            if moves is None or moves[batch] == 1:
                self.tape[batch][self.ptr[batch]] = values[batch]

    def move_ptr(self, move):

        for batch in range(self.ptr.shape[0]):
            new_ptr = self.ptr[batch] + move[batch]
            if new_ptr >= self.tape.shape[1]:
                new_ptr = self.tape.shape[1] - 1
            elif new_ptr < 0:
                new_ptr = 0

            self.ptr[batch] = new_ptr

    def get_ptr(self):
        return self.ptr

    def index_to_moves(self, index):

        moves = np.zeros(index.shape, dtype=np.int32)
        for batch in range(index.shape[0]):
            moves[batch] = self.table[index[batch]]

        return moves

    def print_tape(self):

        print("Tape status : ")

        for i in range(self.tape.shape[0]):
            print("Batch {:2d}".format(i))
            print(self.tape[i])

    def print_max_indexes(self):

        print("Tape max indexes : ")

        for i in range(self.tape.shape[0]):
            print(np.argmax(self.tape[i], axis=1))
