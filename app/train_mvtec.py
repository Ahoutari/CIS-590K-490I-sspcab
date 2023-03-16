
import os
import argparse
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

import json

def parse_user_arguments():
    """Parse user arguments for the evaluation of a method on the MVTec AD
    dataset.

    Returns:
        Parsed user arguments.
    """
    parser = argparse.ArgumentParser(description="""Parse user arguments.""")

    parser.add_argument('--model', required=True, help="""Path to the model to train. Must be a .py file.""")
    parser.add_argument('--out', required=True, help="""Path to the output directory.""")
    
    parser.add_argument('--epochs', type=int, default=20, help="""Number of epochs to train the model.""")
    parser.add_argument('--batch_size', type=int, default=32, help="""Batch size to use during training.""")    
    parser.add_argument('--num_workers', type=int, default=4, help="""Number of workers to use during training.""")
    
    
    parser.add_argument('--dataset', required=True, help="""Path to the dataset to train on. Must follow the MVTec AD format.""")

    args = parser.parse_args()
    
    return args

def main():
    parser = parse_user_arguments()
    
    hyper_params = {
        'epochs': parser.epochs,
        'batch_size': parser.batch_size,
        'workers': parser.num_workers,
    }
    
    base_url = parser.dataset
    
    try:
        model: Sequential = __import__(parser.model)

        model = model.model
    except:
        raise Exception('Could not import model')
    
    SIZE = 128

    batch_size = hyper_params['batch_size']
    datagen = ImageDataGenerator(rescale=1./255)

    train_generator = datagen.flow_from_directory(
        f'{base_url}/train/',
        target_size=(SIZE, SIZE),
        batch_size=batch_size,
        class_mode='input'
        )
    
    history = model.fit(
        train_generator, 
        epochs=hyper_params['epochs'],
        steps_per_epoch=train_generator.samples // batch_size,
        shuffle=True,
        workers=hyper_params['workers'],
        )
    
    model.save(f'{parser.out}/model.h5')
    
    with open(f'{parser.out}/history.json', 'w') as f:
        json.dump(history.history, f)
        
    
if __name__ == '__main__':
    main()
        