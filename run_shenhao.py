import torch
import json
import torch.optim as optim
import time
import os
import pickle
from typing import List

from Skip_gram import SkipGram
from batch_generator import Batch_Generator
from pre_load import CorpusProc


def is_better(loss_list:List[float], loss:float, scale=2):
    """

    :param loss_list:
    :param loss:
    :return:
    """
    loss_list = sorted(loss_list, reverse=True)
    if loss < loss_list[-1]:
        loss_list[0] = loss
        return True, loss_list
    elif loss > loss_list[0] * scale:
        # when the loss is larger than the largest one in k times
        return False, loss_list
    else:
        loss_list = sorted(loss_list.append(loss))
        return True, loss_list[0:-1]




def train():
    """
    use
    :return:
    """
    BATCH_SIZE = 2048

    TRAIN_CORPUS = "train_data_shenhao"
    VOCAB_PATH = os.path.join(".", TRAIN_CORPUS,"vocab.json")
    STATE_PATH = os.path.join(".",TRAIN_CORPUS , "model", "state.json")
    CORPUS_PATH = os.path.join(".", TRAIN_CORPUS, TRAIN_CORPUS)
    model_save_basepath = os.path.join(".", TRAIN_CORPUS, "model")
    CLIP_GRAD = 100
    NUMBER_LATEST_MODEL = 10
    ## the training state(all need to save)
    PRINT_EVERY = 500
    SAVE_EVERY = 1500
    epoch = 1
    epochs = 10
    iter = 0

    ##
    embedding_dim = 100
    WINDOW_SIZE = 2

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    device = torch.device(device)

    # init the params


    ## init the vocab

    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    ## init the corpus
    # corpus_file_list = os.listdir(CORPUS_PATH)
    preprocess = CorpusProc(vocab, VOCAB_PATH)

    # for file in corpus_file_list:
    #     print("Loading the {}".format(file))
    #     filepath = os.path.join(CORPUS_PATH, file)
    #     corpus_temp = preprocess.read_corpus(filepath)
    #     corpus.extend(corpus_temp)

    ## model save params
    model_save_basename = "model"

    model_name_list = []  # need to save

    current_model_point = 0  # need to save
    loss_list = []

    # inistaniating the model

    model = SkipGram(embedding_dim, preprocess.vocab_size, window_size=WINDOW_SIZE)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters())

    print("the vocabulary size is %d." % len(vocab))
    train_time = begin_time = time.time()
    ## init the params if it has been trained
    if os.path.exists(STATE_PATH):
        print("Init the model that has been train.")
        with open(STATE_PATH, 'rb') as f:
            state = pickle.load(f)
            epoch = state['epoch']
            iter = state['iter']
            begin_time = state['begin_time']
            current_model_point = state['current_model_point']
            model_name_list = state['model_name_list']
            loss_list = state['loss_list']

            ### read model
            model_path = model_name_list[(current_model_point - 1) % NUMBER_LATEST_MODEL]
            # params = torch.load(model_path , map_location=lambda storage, loc: storage)
            # model.load_state_dict(torch.load(model_path))
            model = torch.load(model_path)
            optimizer.load_state_dict(torch.load(model_path + '.optim'))


    pre_iter = iter
    print("Now loading the corpus")
    corpus = preprocess.read_corpus(CORPUS_PATH)

    print("Finish loading !")

    print("Begin Training skip-gram")
    if epoch == 0:
        epoch = 1

    while epoch <= epochs:
        #
        print("Not it's {}th epoch.".format(epoch))

        generator = Batch_Generator(corpus, vocab)
        batch_generator = generator.batch_generator(BATCH_SIZE)

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

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)

            optimizer.step()

            # show the state of model and save the model
            if iter % PRINT_EVERY == 0:
                # show the state of training
                print("Now it's epoch %d , iter %d. Now the loss is %.3f."
                      "speed %.2f iter/sec, time elapsed %.2f sec." % (epoch, iter, loss,
                                                                       (iter - pre_iter) / (time.time() - train_time),
                                                                       time.time() - begin_time))
                train_time = time.time()
                pre_iter = iter

            ## save the latest model
            if iter % SAVE_EVERY == 0:
                save_or_not = True
                # if len(loss_list) < NUMBER_LATEST_MODEL:
                #     loss_list.append(loss)
                #
                # else:
                #     save_or_not, loss_list = is_better(loss_list, loss)

                save_or_not = True

                if save_or_not:
                    # print("Now it's epoch %d , iter %d. "
                    #       "The train loss %.3f is better than before."
                    #       "We save the model at direction ./model" % (epoch, iter, loss))

                    model_save_path = os.path.join(model_save_basepath, model_save_basename + str(iter) + '.bin')
                    if len(model_name_list) < NUMBER_LATEST_MODEL:
                        model_name_list.append(model_save_path)

                    else:
                        # print(current_model_point)
                        remove_path = model_name_list[current_model_point]
                        # remove_path = os.path.join(model_save_basepath, remove_path)
                        os.remove(remove_path)
                        os.remove(remove_path + ".optim")
                        model_name_list[current_model_point] = model_save_path
                    current_model_point = (current_model_point + 1) % NUMBER_LATEST_MODEL

                    torch.save(model, model_save_path)
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')

                    ## save the state of model

                    with open(STATE_PATH, 'wb') as f:
                        state = {}
                        state['epoch'] = epoch
                        state['iter'] = iter
                        state['begin_time'] = begin_time
                        state['current_model_point'] = current_model_point
                        state['model_name_list'] = model_name_list
                        state['loss_list'] = loss_list
                        pickle.dump(state, f)

        epoch += 1




if __name__ == '__main__':
    train()


