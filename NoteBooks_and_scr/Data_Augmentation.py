import cv2
import time
import numpy as np
from os import listdir
from keras.preprocessing.image import ImageDataGenerator

# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m}:{round(s, 1)}"

def augment_data(file_dir, n_generated_samples, save_to_dir):
    data_gen = ImageDataGenerator(
        #rotation_range=15,
        brightness_range=(0.8, 1.2),
        horizontal_flip=True,
        vertical_flip=True
    )
    
    for filename in listdir(file_dir):
        # Load the image
        image = cv2.imread(file_dir + '/' + filename)
        # Prefix of the names for the generated samples
        save_prefix = 'aug_' + filename[:-4]  # Remove the file extension
        # Generate 'n_generated_samples' sample images
        i = 0
        for batch in data_gen.flow(x=np.expand_dims(image, axis=0), batch_size=1, save_to_dir=save_to_dir,
                                   save_prefix=save_prefix, save_format='png'):
            i += 1
            if i >= n_generated_samples:
                break

def data_summary(main_path):
    no_dr_path = main_path + 'No_DR'
    moderate_path = main_path + 'Moderate'
    severe_path = main_path + 'Severe'
    
    m_pos = len(listdir(moderate_path))
    m_neg = len(listdir(no_dr_path))
    m_severe = len(listdir(severe_path))
    
    m = m_pos + m_neg + m_severe
    
    pos_prec = (m_pos * 100.0) / m
    neg_prec = (m_neg * 100.0) / m
    severe_prec = (m_severe * 100.0) / m
    
    print(f"Number of examples: {m}")
    print(f"Percentage of positive examples: {pos_prec}%, number of moderate examples: {m_pos}")
    print(f"Percentage of negative examples: {neg_prec}%, number of neg examples: {m_neg}") 
    print(f"Percentage of severe examples: {severe_prec}%, number of severe examples: {m_severe}")

start_time = time.time()

augmented_data_path = 'D:/Advanced AI/Project/augmented data/'

#augment_data(file_dir='D:/Advanced AI/Project/colored_images/Severe/', n_generated_samples=9, save_to_dir=augmented_data_path + 'Severe')
#augment_data(file_dir='D:/Advanced AI/Project/colored_images/No_DR/', n_generated_samples=2, save_to_dir=augmented_data_path + 'No_DR')
augment_data(file_dir='D:/Advanced AI/Project/colored_images/Moderate/', n_generated_samples=4, save_to_dir=augmented_data_path + 'Moderate')

end_time = time.time()
execution_time = end_time - start_time
print(f"Elapsed time: {hms_string(execution_time)}")

data_summary(augmented_data_path)
