from helpers import parameter_parser, tab_printer
from hashing_machine import DistributedHashingMachine

def main(args):
    """
    Main function to read the graph list, extract features, learn the embedding and save it.
    :param args: Object with the arguments.
    """
    model = DistributedHashingMachine(args)
    model.execute_hashing()
    model.save_embedding()

if __name__ == "__main__":
    args = parameter_parser()
    tab_printer(args)
    main(args)
