import argparse
from texttable import Texttable

def parameter_parser():

    """
    A method to parse up command line parameters. By default it gives an embedding of the partial NCI1 graph dataset.
    The default hyperparameters give a good quality representation without grid search.
    Representations are sorted by ID.
    """

    parser = argparse.ArgumentParser(description = "Run Nested Subtree Hashing.")


    parser.add_argument('--input-path',
                        nargs = '?',
                        default = './dataset/',
	                help = 'Input folder with jsons.')

    parser.add_argument('--output-path',
                        nargs = '?',
                        default = './features/nci1.csv',
	                help = 'Embeddings path.')

    parser.add_argument('--dimensions',
                        type = int,
                        default = 16,
	                help = 'Number of dimensions. Default is 16.')

    parser.add_argument('--workers',
                        type = int,
                        default = 4,
	                help = 'Number of workers. Default is 4.')

    parser.add_argument('--wl-iterations',
                        type = int,
                        default = 2,
	                help = 'Number of Weisfeiler-Lehman iterations. Default is 2.')
    
    return parser.parse_args()


def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    tab = Texttable() 
    tab.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print(tab.draw())
