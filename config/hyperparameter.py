def hyperparameter_config():
        return [
            {
                'name': 'single',
                'folder_path': './sandbox/single',
                'chunk_size': [50, 100, 150, 200],
                'chunk_overlap': [0, 10, 30, 50],
                'top_k': [10],
                'train_dataset': './dataset/single_train_parsed.json',
                'test_dataset': './dataset/single_test_parsed.json'
            },
    ]