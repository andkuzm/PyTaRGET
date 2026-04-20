class Broken_to_repaired:
    def __init__(self, broken, repaired, test_name, rel_path, log):
        self.broken = broken
        self.repaired = repaired
        self.test_name = test_name
        self.rel_path = rel_path
        self.log = log

    def __eq__(self, other):
        return (self.broken, self.repaired, self.test_name, self.rel_path) == \
               (other.broken, other.repaired, other.test_name, other.rel_path)

    def __hash__(self):
        return hash((self.broken, self.repaired, self.test_name, self.rel_path))