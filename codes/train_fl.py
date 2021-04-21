import os
import tensorflow as tf
from data_loader_FL import DataGenerator as generator_fl
from data_loader import DataGenerator
from models import VAEmodel, lstmKerasModel
from trainers import vaeTrainer
from aggregator import Aggregator
from utils import process_config, create_dirs, get_args, save_config
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config['result_dir'], config['checkpoint_dir'], config['checkpoint_dir_lstm']])
    # save the config in a txt file
    save_config(config)
    # create tensorflow session
    sessions = []
    data = []
    model_vaes = []
    vae_trainers = []
    lstm_models = []
    model_vae_global = VAEmodel(config, "Global")
    sess_global = tf.Session(config=tf.ConfigProto())
    for i in range(1, 5):
        sess = tf.Session(config=tf.ConfigProto())
        sessions.append(sess)
        data.append(generator_fl(config, i))
        model_vaes.append(VAEmodel(config, "Client{}".format(i)))
        model_vaes[-1].load(sessions[-1])
        vae_trainers.append(vaeTrainer(sessions[-1], model_vaes[-1], data[-1], config))
        lstm_models.append(lstmKerasModel("Client{}".format(i), config))
    model_vae_global.load(sess_global)
    trainer_vae_global = vaeTrainer(sess_global, model_vae_global, data[0], config)
    lstm_model_global = lstmKerasModel("Global", config)
    # client_weights = [0.1] * 8
    client_weights = [0.25, 0.25, 0.25, 0.25]
    # client_weights.append(0.2)
    aggregator = Aggregator(vae_trainers, trainer_vae_global, lstm_models, lstm_model_global, config, client_weights)
    aggregator.aggregate_vae()
    aggregator.aggregate_lstm()

if __name__ == '__main__':
    main()
