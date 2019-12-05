import os

model_save_basepath = os.path.join(".", "model")
model_save_basename = "model"
model_name_list = []
num_latest_file = 5
current_model_point = 0


for iter in range(100):
    model_save_path = os.path.join(model_save_basepath, model_save_basename + str(iter) + ".bin")
    if len(model_name_list) < num_latest_file:
        model_name_list.append(model_save_path)

    else:
        print(current_model_point)
        remove_path = model_name_list[current_model_point]

        os.remove(remove_path)
        model_name_list[current_model_point] = model_save_path
    current_model_point = (current_model_point + 1) % num_latest_file

    with open(model_save_path, 'w') as f:
        f.write(str(iter))
