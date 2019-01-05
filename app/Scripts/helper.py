import pandas as pd


def clean_data(file_data):
    f = open(file_data, 'r')
    data = f.read().split("\n")
    result_data = []
    for row in data:
        result_data.append(row.split(','))
    data_frame = pd.DataFrame(result_data,
                              columns=['classes', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
                                       'gill-attachment',
                                       'gill-spacing',
                                       'gill-size', 'gill-color', 'stalk-shape',
                                       'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
                                       'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
                                       'veil-color', 'ring-number', 'ring-type',
                                       'spore-print-color', 'population', 'habitat'], dtype=int)

    # create dict for col
    classes = {'e': -1, 'p': 1}
    cap_shape = {'b': 1, 'c': 2, 'x': 3, 'f': 4, 'k': 5, 's': 6}
    cap_surface = {'f': 1, 'g': 2, 'y': 3, 's': 4}
    cap_color = {'n': 1, 'b': 2, 'c': 3, 'g': 4, 'r': 5, 'p': 6, 'u': 7, 'e': 8, 'w': 9, 'y': 10}
    bruises = {'t': 1, 'f': 0}
    odor = {'a': 1, 'l': 2, 'c': 3, 'y': 4, 'f': 5, 'm': 6, 'n': 7, 'p': 8, 's': 9}
    gill_attachment = {'a': 1, 'd': 2, 'f': 3, 'n': 4}
    gill_spacing = {'c': 1, 'w': 2, 'd': 3}
    gill_size = {'b': 1, 'n': 2}
    gill_color = {'k': 1, 'n': 2, 'b': 3, 'h': 4, 'g': 5, 'r': 6, 'o': 7, 'p': 8, 'u': 9, 'e': 10, 'w': 11, 'y': 12}
    stalk_shape = {'e': 1, 't': 2}
    stalk_root = {'b': 1, 'c': 2, 'u': 3, 'e': 4, 'z': 5, 'r': 6, '?': 7}
    stalk_surface_above_ring = {'f': 1, 'y': 2, 'k': 3, 's': 4}
    stalk_surface_below_ring = {'f': 1, 'y': 2, 'k': 3, 's': 4}
    stalk_color_above_ring = {'n': 1, 'b': 2, 'c': 3, 'g': 4, 'o': 5, 'p': 6, 'e': 7, 'w': 8, 'y': 9}
    stalk_color_below_ring = {'n': 1, 'b': 2, 'c': 3, 'g': 4, 'o': 5, 'p': 6, 'e': 7, 'w': 8, 'y': 9}
    veil_type = {'p': 1, 'u': 2}
    veil_color = {'n': 1, 'o': 2, 'w': 3, 'y': 4}
    ring_number = {'n': 1, 'o': 2, 't': 3}
    ring_type = {'c': 1, 'e': 2, 'f': 3, 'l': 4, 'n': 5, 'p': 6, 's': 7, 'z': 8}
    spore_print_color = {'k': 1, 'n': 2, 'b': 3, 'h': 4, 'r': 5, 'o': 6, 'u': 7, 'w': 8, 'y': 9}
    population = {'a': 1, 'c': 2, 'n': 3, 's': 4, 'v': 5, 'y': 6}
    habitat = {'g': 1, 'l': 2, 'm': 3, 'p': 4, 'u': 5, 'w': 6, 'd': 7}
    # Convert alpha to int
    convert_alpha_to_int(data_frame, 'classes', classes)
    convert_alpha_to_int(data_frame, 'cap-shape', cap_shape)
    convert_alpha_to_int(data_frame, 'cap-surface', cap_surface)
    convert_alpha_to_int(data_frame, 'cap-color', cap_color)
    convert_alpha_to_int(data_frame, 'bruises', bruises)
    convert_alpha_to_int(data_frame, 'odor', odor)
    convert_alpha_to_int(data_frame, 'gill-attachment', gill_attachment)
    convert_alpha_to_int(data_frame, 'gill-spacing', gill_spacing)
    convert_alpha_to_int(data_frame, 'gill-size', gill_size)
    convert_alpha_to_int(data_frame, 'gill-color', gill_color)
    convert_alpha_to_int(data_frame, 'stalk-shape', stalk_shape)
    convert_alpha_to_int(data_frame, 'stalk-root', stalk_root)
    convert_alpha_to_int(data_frame, 'stalk-surface-above-ring', stalk_surface_above_ring)
    convert_alpha_to_int(data_frame, 'stalk-surface-below-ring', stalk_surface_below_ring)
    convert_alpha_to_int(data_frame, 'stalk-color-above-ring', stalk_color_above_ring)
    convert_alpha_to_int(data_frame, 'stalk-color-below-ring', stalk_color_below_ring)
    convert_alpha_to_int(data_frame, 'veil-type', veil_type)
    convert_alpha_to_int(data_frame, 'veil-color', veil_color)
    convert_alpha_to_int(data_frame, 'ring-number', ring_number)
    convert_alpha_to_int(data_frame, 'ring-type', ring_type)
    convert_alpha_to_int(data_frame, 'spore-print-color', spore_print_color)
    convert_alpha_to_int(data_frame, 'population', population)
    convert_alpha_to_int(data_frame, 'habitat', habitat)

    return data_frame


def convert_alpha_to_int(data_frame, colume, dick_convert):
    list_data_colume = list(data_frame[colume])
    for k, v in dick_convert.items():
        for index, bruise in enumerate(list_data_colume):
            if bruise == k:
                list_data_colume[index] = v

    data_frame[colume] = list_data_colume


def x_train_y_train(data_frame):
    train_data = data_frame[:-1].astype(int)
    x_train = train_data.drop('classes', 1).values
    y_train = train_data['classes'].values
    return x_train, y_train


def X_train_Y_train_X_test_Y_test(data_frame):
    train_data = data_frame[:6001].astype(int)
    test_data = data_frame[6001:len(data_frame) - 1].astype(int)
    X_train = train_data.drop('classes', 1).values
    Y_train = train_data['classes'].values
    X_test = test_data.drop('classes', 1).values
    Y_test = test_data['classes'].values
    return X_train, Y_train, X_test, Y_test
