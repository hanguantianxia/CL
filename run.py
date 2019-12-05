import torch
import json
import torch.optim as optim
import time
import os
import pickle

from Skip_gram import SkipGram
from batch_generator import Batch_Generator
from pre_load import CorpusPreprocess




def train():
    """
    use
    :return:
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    device = torch.device(device)
    state_path = os.path.join(".", "state.json")
    # init the params


    ## init the vocab
    vocab_path = "vocab.json"
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    ## init the corpus
    corpus_path = os.path.join(".", "train_data")
    corpus_file_list = os.listdir(corpus_path)
    preprocess = CorpusPreprocess(vocab)
    corpus = []
    for file in corpus_file_list:
        print("Loading the {}".format(file))
        filepath = os.path.join(corpus_path, file)
        corpus_temp = preprocess.read_corpus(filepath)
        corpus.extend(corpus_temp)

    ## model save params
    BATCH_SIZE = 2048
    model_save_basepath = os.path.join(".", "model")
    model_save_basename = "model"
    model_name_list = []  # need to save
    num_latest_file = 5
    current_model_point = 0  # need to save

    # inistaniating the model
    embedding_dim = 100
    window_size = 2
    model = SkipGram(embedding_dim, preprocess.vocab_size, window_size=2)

    optimizer = optim.Adam(model.parameters())

    ## the training state(all need to save)
    print_every = 1500
    epoch = 0
    epochs = 5
    iter = 0
    train_time = begin_time = time.time()




    ## init the params if it has been trained
    if os.path.exists(state_path):
        print("Init the model that has been train.")
        with open(state_path, 'rb') as f:
            state = pickle.load(f)
            epoch = state['epoch']
            iter = state['iter']
            begin_time = state['begin_time']
            current_model_point = state['current_model_point']
            model_name_list = state['model_name_list']

            ### read model
            model_path = model_name_list[(current_model_point - 1) % num_latest_file]
            # params = torch.load(model_path , map_location=lambda storage, loc: storage)
            # model.load_state_dict(torch.load(model_path))
            model = torch.load(model_path)
            optimizer.load_state_dict(torch.load(model_path + '.optim'))

    model = model.to(device)

    print("Begin Training skip-gram")

    while epoch < epochs:
        #
        print("Not it's {}th epoch.".format(epoch+1))
        epoch += 1
        generator = Batch_Generator(corpus, vocab)
        batch_generator = generator.batch_generator(1024)

        for input_vectors,output_vectors,noise_vectors in batch_generator:
            iter += 1
            #
            # ### training code
            input_vectors = input_vectors.to(device)
            output_vectors = output_vectors.to(device)
            noise_vectors = noise_vectors.to(device)
            # get train vecotr
            loss = model(input_vectors, output_vectors, noise_vectors)

            # optimize the model
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            # show the state of model and save the model
            if iter % print_every == 0:
                # show the state of training
                print("Now it's %d epoch, %d iter. Now the loss is %.2f."
                      "speed %.2f iter/sec, time elapsed %.2f sec." % (epoch, iter, loss,
                                                                       iter / (time.time() - begin_time),
                                                                       time.time() - begin_time))
                train_time = time.time()

                ## save the latest model
                model_save_path = os.path.join(model_save_basepath, model_save_basename + str(iter) + '.bin')
                if len(model_name_list) < num_latest_file:
                    model_name_list.append(model_save_path)

                else:
                    print(current_model_point)
                    remove_path = model_name_list[current_model_point]

                    os.remove(remove_path)
                    model_name_list[current_model_point] = model_save_path
                current_model_point = (current_model_point + 1) % num_latest_file

                torch.save(model, model_save_path)
                torch.save(optimizer.state_dict(), model_save_path + '.optim')

                ## save the state of model

                with open(state_path, 'wb') as f:
                    state = {}
                    state['epoch'] = epoch
                    state['iter'] = iter
                    state['begin_time'] = begin_time
                    state['current_model_point'] = current_model_point
                    state['model_name_list'] = model_name_list
                    pickle.dump(state, f)






if __name__ == '__main__':
    train()


