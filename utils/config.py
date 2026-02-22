import yaml
import argparse

"""
Author: Helena Russello
"""

class MyConfig(object):

    def __init__(self, args):
        """
         A custom class for reading and parsing a YAML configuration file.

        :param config_path: the path of the configuration file
        """
        config_path = args.config

        self.config = None
        with open(config_path) as f:
            # load config file
            self.config = yaml.load(f, Loader=yaml.SafeLoader)
            self.fold_id = args.fold_id

        self.wandb = self.config['wandb']

        # data fields
        self.gait_scores_csv = self.config['gait_scores_csv']
        self.keypoints_path = self.config['keypoints_path']
        self.merging = self.config['merging']
        self.n_folds = self.config['n_folds']
        self.use_kp = self.config['use_kp']
        self.flat_cv = self.config['flat_cv']

        # model fields
        self.load_model = self.config['load_model']
        self.save_path = self.config['save_path']
        self.model_name = self.config['model_name']
        self.model_params = self.config['model_params']

        # training fields
        self.seed = self.config['seed']
        self.lr = float(self.config['lr'])
        self.optimizer = self.config['optimizer']
        self.batch_size = self.config['batch_size']
        self.epochs = self.config['epochs']
        self.seq_length = self.config['seq_length']
        self.step_size = self.config['step_size']
        self.gradient_clipping = float(self.config['gradient_clipping'])
        self.weight_decay = float(self.config['weight_decay'])
        self.sweep = self.config['sweep']
        self.jitter_percentage = self.config['jitter_percentage']

        if args.model_type:
            self.model_type = args.model_type
        if args.load_model:
            self.load_model = args.load_model
        if args.merging:
            self.merging = args.merging
        if args.flat_cv:
            self.flat_cv = args.flat_cv

        if args.batch_size:
            self.batch_size = args.batch_size
        if args.epochs:
            self.epochs = args.epochs
        if args.gradient_clipping:
            self.gradient_clipping = args.gradient_clipping
        if args.lr:
            self.lr = args.lr
        if args.bidirectional:
            self.model_params['bidirectional'] = args.bidirectional
        if args.dropout:
            self.model_params['dropout'] = args.dropout
        if args.num_hidden:
            self.model_params['num_hidden'] = args.num_hidden
        if args.num_layers:
            self.model_params['num_layers'] = args.num_layers
        if args.step_size:
            self.step_size = args.step_size
        if args.optimizer:
            self.optimizer = args.optimizer
        if args.seq_length:
            self.seq_length = args.seq_length
        if args.weight_decay:
            self.weight_decay = args.weight_decay
        if args.jitter_percentage:
            self.jitter_percentage = args.jitter_percentage

        self.config = config_path

    def __str__(self):
        return str(self.config)


def parse_args(description):
    """
    Parse arguments and process the configuration file
    :return: the config and the arguments
    """
    parser = argparse.ArgumentParser(description=description)
    # config file
    parser.add_argument('--config',
                        help='YAML configuration file',
                        default="cfg/config.yml",
                        type=str)

    parser.add_argument('--model_type',
                        help='Classifier type',
                        type=str)
    parser.add_argument('--load_model',
                        help='Model to load',
                        type=str)
    parser.add_argument('--merging',
                        help='How to merge the scores',
                        type=str)

    parser.add_argument('--flat_cv',
                        help='Whether to perform flat or nested cv',
                        type=str)
    ## wandb sweep
    parser.add_argument('--batch_size',
                        help='',
                        type=int)
    parser.add_argument('--bidirectional',
                        help='',
                        type=bool)
    parser.add_argument('--dropout',
                        help='',
                        type=float)
    parser.add_argument('--epochs',
                        help='',
                        type=int)
    parser.add_argument('--gradient_clipping',
                        help='',
                        type=float)
    parser.add_argument('--lr',
                        help='',
                        type=float)
    parser.add_argument('--num_hidden',
                        help='',
                        type=int)
    parser.add_argument('--num_layers',
                        help='',
                        type=int)
    parser.add_argument('--optimizer',
                        help='',
                        type=str)
    parser.add_argument('--seq_length',
                        help='',
                        type=int)
    parser.add_argument('--weight_decay',
                        help='',
                        type=float)
    parser.add_argument('--step_size',
                        help='',
                        type=int)
    parser.add_argument('--jitter_percentage',
                        help='',
                        type=float)

    parser.add_argument('--fold_id',
                        help='run only fold x',
                        default=-1,
                        type=int)

    args, rest = parser.parse_known_args()
    print(args)

    cfg = MyConfig(args)
    print(cfg)

    args = parser.parse_args()  # parse the rest of the args

    return cfg, args
