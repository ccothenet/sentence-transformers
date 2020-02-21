from . import InputExample


class NLIDataReader(object):
    """
    Reads in the Stanford NLI dataset and the MultiGenre NLI dataset
    """
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder

    def get_examples(self, dataset, max_examples=0):
        """
        """

        with open(self.dataset_folder + "/fr.raw." + dataset, "r", encoding="utf-8") as rfile:
            lines = rfile.readlines()[1:]
        s1 = [o.split("\t")[0].strip() for o in lines]
        s2 = [o.split("\t")[1].strip() for o in lines]
        labels = [o.split("\t")[2].strip() for o in lines]
        examples = []
        id = 0
        for sentence_a, sentence_b, label in zip(s1, s2, labels):
            guid = "%s-%d" % (dataset, id)
            id += 1
            examples.append(InputExample(guid=guid, texts=[sentence_a, sentence_b], label=self.map_label(label)))

            if 0 < max_examples <= len(examples):
                break

        return examples

    @staticmethod
    def get_labels():
        return {"contradiction": 0, "entailment": 1, "neutral": 2}

    def get_num_labels(self):
        return len(self.get_labels())

    def map_label(self, label):
        return self.get_labels()[label.strip().lower()]