import torch
import json
import torch.optim as optim
import time
import os
import pickle

from Skip_gram import SkipGram


def train():
    """
    use
    :return:
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    state_path = os.path.join(".", "state.json")
    # init the params
    vocab = json

    ## model save params
    model_save_basepath = os.path.join(".", "model")
    model_save_basename = "model"
    model_name_list = []  # need to save
    num_latest_file = 5
    current_model_point = 0  # need to save

    # inistaniating the model
    embedding_dim = 100
    n_vocab = len(vocab)
    model = SkipGram(embedding_dim, n_vocab)

    optimizer = optim.Adam(model.parameters())

    ## the training state(all need to save)
    print_every = 1500
    epoch = 0
    epochs = 5
    iter = 0
    train_time = begin_time = time.time()

    # train for some number of epochs

    ## init the params if it has been trained
    if os.path.exists(state_path):
        with open(state_path, 'rb') as f:
            state = pickle.load(f)
            epoch = state['epoch']
            iter = state['iter']
            begin_time = state['begin_time']
            current_model_point = state['current_model_point']
            model_name_list = state['model_name_list']

            ### read model
            model_path = model_name_list[(current_model_point - 1) % num_latest_file]
            params = torch.load(model_path + '.bin', map_location=lambda storage, loc: storage)
            model.load_state_dict(params['state_dict'])
            optimizer.load_state_dict(torch.load(model_path + '.optim'))

    model = model.to(device)

    while epoch < epochs:
        #
        print("Not it's {}th epoch.".format(epoch))

        iter += 1

        ### training code

        # get train vecotr
        input_vectors = None
        output_vectors = None
        noise_vectors = None
        loss = model.forward(input_vectors, output_vectors, noise_vectors)

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

            model.save(model_save_path)
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









