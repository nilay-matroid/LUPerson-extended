from datasets.last import load_LaST
from datasets.market1501 import load_Market1501
from datasets.sysu30k import load_sysu30k


def load_accumulated_info_of_dataset(root_folder_path, dataset_name, **kwargs):
    print("Use {} as root_folder_path ...".format(root_folder_path))

    dataset_name_to_load_function_dict = {
        "Market1501": load_Market1501,
        "LaST": load_LaST,
        "SYSU-30k": load_sysu30k,
    }
    assert dataset_name in dataset_name_to_load_function_dict
    load_function = dataset_name_to_load_function_dict[dataset_name]
    train_and_valid_accumulated_info_dataframe, test_query_accumulated_info_dataframe, test_gallery_accumulated_info_dataframe = load_function(
        root_folder_path=root_folder_path, **kwargs)

    return train_and_valid_accumulated_info_dataframe, test_query_accumulated_info_dataframe, test_gallery_accumulated_info_dataframe