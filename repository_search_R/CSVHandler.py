import csv
import os


class CSVHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.columns = ["SHTCS_T", "SHTCS_P", "TRVME_T", "TRVME_P"]
        self._initialize_csv()

    def _initialize_csv(self):
        if not os.path.exists(self.file_path) or os.stat(self.file_path).st_size == 0:
            with open(self.file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(self.columns)
                writer.writerow([0] * len(self.columns))

    def _increment_column(self, col_index):
        with open(self.file_path, mode='r', newline='') as file:
            reader = list(csv.reader(file))

        reader[1][col_index] = str(int(reader[1][col_index]) + 1)

        with open(self.file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(reader)

    def increment_shtcs_t(self):
        self._increment_column(0)

    def increment_shtcs_p(self):
        self._increment_column(1)

    def increment_trvme_t(self):
        self._increment_column(2)

    def increment_trvme_p(self):
        self._increment_column(3)