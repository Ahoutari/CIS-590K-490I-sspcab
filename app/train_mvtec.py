
import os
import argparse
from keras.preprocessing.image import ImageDataGenerator
from sklearn.neighbors import KernelDensity
import numpy as np
import pickle

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
    
    validation_generator = datagen.flow_from_directory(
        f'{base_url}/test/',
        target_size=(128, 128),
        batch_size=32,
        class_mode='input',
        classes=['good']
        )

    anomaly_generator = datagen.flow_from_directory(
        f'{base_url}/test/',
        target_size=(128, 128),
        batch_size=32,
        class_mode='input',
        classes=['defective']
        )
    
    history = model.fit(
        train_generator, 
        epochs=hyper_params['epochs'],
        steps_per_epoch=train_generator.samples // batch_size,
        shuffle=True,
        workers=hyper_params['workers'],
        )
    
    validation_error = model.evaluate(validation_generator)
    anomaly_error = model.evaluate(anomaly_generator)
    
    model.save(f'{parser.out}/model.h5')

    feature_extractor = Sequential()

    for layer in model.layers[:5]:
        feature_extractor.add(layer)
        
    feature_extractor.save(f'{parser.out}/feature_extractor.h5')    
    
    #Get encoded output of input images = Latent space
    encoded_images = feature_extractor.predict(validation_generator)

    # Flatten the encoder output because KDE from sklearn takes 1D vectors as input
    encoder_output_shape = feature_extractor.output_shape #Here, we have 16x16x16
    out_vector_shape = encoder_output_shape[1]*encoder_output_shape[2]*encoder_output_shape[3]

    encoded_images_vector = [np.reshape(img, (out_vector_shape)) for img in encoded_images]

    #Fit KDE to the image latent data
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(encoded_images_vector)
    
    # save the model to disk
    filename = f'{parser.out}/kde.pickle'
    
    with open(filename, 'wb') as f:
        pickle.dump(kde, f)
    
    def calc_density_and_recon_error(batch_images):
        density_list=[]
        recon_error_list=[]
        for im in range(0, batch_images.shape[0]-1):
            
            img  = batch_images[im]
            img = img[np.newaxis, :,:,:]
            encoded_img = feature_extractor.predict([[img]]) # Create a compressed version of the image using the encoder
            encoded_img = [np.reshape(img, (out_vector_shape)) for img in encoded_img] # Flatten the compressed image
            density = kde.score_samples(encoded_img)[0] # get a density score for the new image
            reconstruction = model.predict([[img]])
            reconstruction_error = model.evaluate([reconstruction],[[img]], batch_size = 1)[0]
            density_list.append(density)
            recon_error_list.append(reconstruction_error)
            
        average_density = np.mean(np.array(density_list))  
        stdev_density = np.std(np.array(density_list)) 
        
        average_recon_error = np.mean(np.array(recon_error_list))  
        stdev_recon_error = np.std(np.array(recon_error_list)) 
        
        return average_density, stdev_density, average_recon_error, stdev_recon_error
    
    err = {
        'validation_error': validation_error,
        'anomaly_error': anomaly_error
    }
    
    train_batch = validation_generator.next()[0]
    anomaly_batch = anomaly_generator.next()[0]

    normal_values = calc_density_and_recon_error(train_batch)
    anomaly_values = calc_density_and_recon_error(anomaly_batch)
    
    err['normal'] = normal_values
    err['anomaly'] = anomaly_values
    
    
    with open(f'{parser.out}/history.json', 'w') as f:
        json.dump(history.history, f)
        
    with open(f'{parser.out}/metrics.json', 'w') as f:
        json.dump(err, f)
    
        
    
if __name__ == '__main__':
    main()
        