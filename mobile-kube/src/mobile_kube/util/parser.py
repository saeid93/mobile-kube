from random import uniform


class Parser:
    """Abstract class to read files"""
    def __init__(self, file):
        self.line = 0
        self.file = open(file)

    def read_line(self):
        ...

    def __enter__(self):
        return self


class UserParser(Parser):
    """Class to read users data"""
    def __init__(self, file):
        super().__init__(file)
        self.empty = False
        self.users_info = None

    def read_line(self):
        self.line += 1
        context = self.file.readline()

        if not context:
            self.empty = True
            return

        self.users_info_line = context.split()[0]
        self.users_info = self.users_info_line.split(',')
        self.users_info = list(map(float, self.users_info))
        return self.users_info

    def get_user(self, user_id):
        try:
            users_coordinates = \
                self.users_info[user_id*2+1], self.users_info[user_id*2+2]
        except IndexError:
            raise IndexError((f"Requested user_id {user_id} "
                              "does not exists in the dataset. "
                              "The number of users in the dataset exceeds "
                              "the number of users in the dataset. "))
        return users_coordinates

    def serialize(self):
        ...


class NodeParser(Parser):
    """Class to parse nodes data"""
    def __init__(self, file):
        super().__init__(file)
        self.empty = False

    def read_line(self):
        """read a line from nodes dataset"""
        self.line += 1
        context = self.file.readline()

        if not context:
            return -1

        return context.split(',')[1:]

    def is_empty(self):
        """check whether file is empty"""
        return self.empty

    def serialize(self):
        """return all servers coordinates as a list"""
        context = []
        while True:
            line = self.read_line()

            if line == -1:
                break

            line = list(map(float, line))

            context.append(line)

        return context


class StationParser(Parser):
    """Class to parse stations"""
    def __init__(self, file):
        super().__init__(file)
        self.empty = False

    def read_line(self):
        """read a single line"""
        self.line += 1
        context = self.file.readline()

        if not context:
            return -1

        return context.split(',')[1:]

    def is_empty(self):
        """check whether file is empty"""
        return self.empty

    def serialize(self):
        """return all stations coordinates as a list"""
        context = []
        while True:
            line = self.read_line()

            if line == -1:
                break
            line = list(map(float, line))
            # add a small nudge to the dataset for servers
            # line = list(map(lambda x: x + uniform(-0.001, 0.001), line))
            nudge = 0.0001
            line = list(map(lambda x: x + 0.0001, line))
            context.append(line)

        return context


# if __name__ == '__main__':
#     import os
#     datasets_folder = "/Users/saeid/Codes/CCGrid-data-repo/myenv/networks-data/"
#     dataset_id = 0
#     dataset_path = os.path.join(datasets_folder, str(dataset_id))
#     sp = nodeParser(os.path.join(dataset_path, 'nodes.txt'))
#     data = sp.serialize()
#     print(data)

#     sp = StationParser(os.path.join(dataset_path, 'stations.txt'))
#     data = sp.serialize()
#     print(data)

#     up = UserParser(os.path.join(dataset_path, 'users.txt'))
#     while up.empty is not True:
#         row = up.read_line()
#         print(up.get_user(2))

