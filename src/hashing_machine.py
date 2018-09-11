import hashlib
import glob
import json
import pandas as pd
import networkx as nx
from tqdm import tqdm
from joblib import Parallel, delayed

class SubtreeHashingMachine:
    """
    Weisfeiler Lehman feature extractor class.
    """
    def __init__(self, graph, features, args):
        """
        Initialization method which executes feature extraction.
        :param graph: The Nx graph object.
        :param features: Feature hash table.
        :param iterations: Number of WL iterations.
        """
        self.args = args
        self.graph = graph
        self.features = features
        self.create_base_features()
        self.do_recursions()

    def single_hash(self, feature):
        """
        """
        hash_object = hashlib.md5(feature.encode())
        hash_value = hash_object.hexdigest()
        number = int(hash_value,16)
        return number, hash_value


    def hashing_tool(self, scores, feature):
        """
        """
        feature = str(feature)
        index_number, index_hash_value = self.single_hash(feature)
        sign_number, sign_hash_value = self.single_hash(feature + "_")
        index = index_number % self.args.dimensions
        sign = (sign_number % 2)*2 -1.0
        scores[index] = scores[index] + sign
        return scores, index_hash_value


    def create_base_features(self):
        """
        """
        scores = [0.0]*self.args.dimensions
        for feature in self.features:
            scores, hash_value = self.hashing_tool(scores, feature)
        self.scores = scores

    def do_a_recursion(self):
        """
        The method does a single WL recursion. Creates a hash vector with the features.
        :return new_features: The hash table with extracted WL features.
        """
        new_features = {}
        scores = [0.0]*self.args.dimensions
        for node in self.graph.nodes():
            nebs = self.graph.neighbors(node)
            degs = [self.features[neb] for neb in nebs]
            feature = "_".join([str(self.features[node])]+list(set(sorted(map(lambda x: str(x),degs)))))
            scores, hash_value = self.hashing_tool(scores, feature)
            new_features[node] = hash_value
        self.scores = self.scores + scores
        return new_features

    def do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for iteration in range(self.args.wl_iterations):
            self.features = self.do_a_recursion()


def dataset_reader(path):
    """
    Function to read the graph and features from a json file.
    :param path: The path to the graph json.
    :return graph: The graph object.
    :return features: Features hash table.
    :return name: Name of the graph.
    """
    name = path.strip(".json").split("/")[-1]
    data = json.load(open(path))
    graph = nx.from_edgelist(data["edges"])

    if "features" in data.keys():
        features = data["features"]
    else:
        features = nx.degree(graph)

    features = {int(k):v for k,v, in features.iteritems()}
    return graph, features, name

def hash_wrap(path, args):
    """
    Function to extract WL features from a graph.
    :param path: The path to the graph json.
    :param rounds: Number of WL iterations.
    :return doc: Document collection object.
    """
    graph, features, name = dataset_reader(path)
    machine = SubtreeHashingMachine(graph, features, args)
    return [name] + machine.scores
       
class DistributedHashingMachine:
    """
    """
    def __init__(self, args):
        """
        """
        self.args = args
        self.graphs = glob.glob(args.input_path + "*.json")

    def execute_hashing(self):
        """
        Function to create the representations in a parallel setup.
        """
        self.hashes = Parallel(n_jobs = self.args.workers)(delayed(hash_wrap)(g, self.args) for g in tqdm(self.graphs))


    def save_embedding(self):
        """
        Function to save the embedding.
        """
        self.feature_count = self.args.dimensions*(self.args.wl_iterations+1)
        self.column_names = ["name"] + map(lambda x: "x_" +str(x), range(0,self.feature_count))

        self.hashes = pd.DataFrame(self.hashes, columns = self.column_names)
        self.hashes = self.hashes.sort_values(["name"])
        self.hashes.to_csv(self.args.output_path, index = None)

