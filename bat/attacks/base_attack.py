class BaseAttack():
    def __init__(self,  classifier):
        """
        Initialize the attack.
        :param classifier: model to attack
        """
        self.classifier = classifier
        self.classifier.n_targets = 0

    def init(self, x):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def batch(self):
        raise NotImplementedError

    def attack(self):
        raise NotImplementedError
