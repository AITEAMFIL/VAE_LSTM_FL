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
    # sessions = []
    # data = []
    # model_vaes = []
    # vae_trainers = []
    # lstm_models = []
    # model_vae_global = VAEmodel(config, "Global")
    sess_global = tf.Session(config=tf.ConfigProto())
    # for i in range(1, 10):
    #     sess = tf.Session(config=tf.ConfigProto())
    #     sessions.append(sess)
    #     data.append(generator_fl(config, i))
    #     model_vaes.append(VAEmodel(config, "Client{}".format(i)))
    #     model_vaes[-1].load(sessions[-1])
    #     vae_trainers.append(vaeTrainer(sessions[-1], model_vaes[-1], data[-1], config))
    #     lstm_models.append(lstmKerasModel("Client{}".format(i), config))
    # model_vae_global.load(sess_global)
    # trainer_vae_global = vaeTrainer(sess_global, model_vae_global, data[0], config)
    # lstm_model_global = lstmKerasModel("Global", config)
    # client_weights = [0.1] * 8
    # client_weights.append(0.2)
    # aggregator = Aggregator(vae_trainers, trainer_vae_global, lstm_models, lstm_model_global, config, client_weights)
    # aggregator.aggregate_vae()
    # aggregator.aggregate_lstm()
    data = DataGenerator(config)
    model_vae = VAEmodel(config, "Global")
    model_vae.load(sess_global)
    trainer_vae = vaeTrainer(sess_global, model_vae, data, config)
    # here you train your model
    if config['TRAIN_VAE']:
        if config['vae_epochs_per_comm_round'] > 0:
            trainer_vae.train()

    if config['TRAIN_LSTM']:
        # create a lstm model class instance
        lstm_model = lstmKerasModel("Global", config)

        # produce the embedding of all sequences for training of lstm model
        # process the windows in sequence to get their VAE embeddings
        lstm_model.produce_embeddings(model_vae, data, sess_global)

        # Create a basic model instance
        lstm_nn_model = lstm_model.lstm_nn_model
        lstm_nn_model.summary()   # Display the model's architecture
        # checkpoint path
        checkpoint_path = lstm_model.config['checkpoint_dir_lstm']\
                                        + "cp_{}.ckpt".format(lstm_model.name)
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                          save_weights_only=True,
                                                          verbose=1)
        # load weights if possible
        # lstm_model.load_model(lstm_nn_model, config, checkpoint_path)

        # start training
        if config['lstm_epochs_per_comm_round'] > 0:
            lstm_model.train(lstm_nn_model, cp_callback)

        # make a prediction on the test set using the trained model
        # lstm_embedding = lstm_nn_model.predict(lstm_model.x_test, batch_size=config['batch_size_lstm'])
        # print(lstm_embedding.shape)

        # # visualise the first 10 test sequences
        # for i in range(10):
        #     lstm_model.plot_lstm_embedding_prediction(i, config, model_vae, sess, data, lstm_embedding)
    sess_global.close()


if __name__ == '__main__':
    main()
