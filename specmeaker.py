import os
import numpy as np
import matplotlib.pyplot as plt
import pywt
import torch
import scipy.io

def footstep_concatenation(x_train, y_train, footsteps_num):
    """
    Concatenates multiple footsteps into a single training sample.

    Parameters:
    - x_train: np.array
        Input feature matrix of shape (samples, features).
    - y_train: np.array
        Corresponding labels (e.g., participant IDs).
    - footsteps_num: int
        Number of footsteps to concatenate into one sample.

    Returns:
    - dict with keys:
        'data_set': torch.Tensor of concatenated input data.
        'labels_set': torch.Tensor of corresponding labels.
    """

    trn_smpl = x_train.shape[0]
    nm_trn = (trn_smpl // footsteps_num) * footsteps_num
    y_train = y_train[:nm_trn]

    trn_idx = np.arange(nm_trn)
    trn_rsz_nm = nm_trn // footsteps_num
    trn_resized = trn_idx.reshape(trn_rsz_nm, footsteps_num)

    # Prepare labels matrix to check consistency
    y_trn_comb = np.zeros([trn_rsz_nm, footsteps_num])
    for col in range(footsteps_num):
        y_trn_comb[:, col] = y_train[trn_resized[:, col]]

    # Remove rows with mixed labels
    mismatched_elmnts = [
        row_idx for row_idx, row in enumerate(y_trn_comb)
        if len(np.unique(row)) > 1
    ]
    y_trn_comb = np.delete(y_trn_comb, mismatched_elmnts, axis=0)
    trn_resized = np.delete(trn_resized, mismatched_elmnts, axis=0)

    # Concatenate footsteps horizontally
    x_train = torch.Tensor(x_train)
    train_set = x_train[trn_resized[:, 0]]
    for i in range(1, footsteps_num):
        train_set = torch.cat((train_set, x_train[trn_resized[:, i]]), dim=1)

    # Use the first label from consistent rows
    y_train = torch.Tensor(y_trn_comb[:, 0]).type(torch.LongTensor)

    return {'data_set': train_set, 'labels_set': y_train}


def generate_cwt_images(file_path, output_path, footsteps_num=1, scales=np.arange(1, 257), wavelet='morl'):
    """
    Loads footstep data, concatenates footsteps, applies CWT, and saves the resulting images.

    Parameters:
    - file_path: str
        Path to the .mat input file containing footstep features.
    - output_path: str
        Directory to save the CWT image outputs.
    - footsteps_num: int
        Number of footsteps to concatenate per sample.
    - scales: np.array
        Scales used for the wavelet transform.
    - wavelet: str
        Wavelet type (e.g., 'morl', 'cmor').
    """

    # === Load dataset from .mat file ===
    dataset = scipy.io.loadmat(file_path)
    footstep_dataset = dataset['footstep_feat']
    features = footstep_dataset[:, :-1]         # All columns except last = features
    labels = footstep_dataset[:, -1] - 1        # Last column = labels (0-indexed)

    # === Concatenate footsteps ===
    data = footstep_concatenation(features, labels, footsteps_num)
    X, y = data['data_set'], data['labels_set']

    # === Generate CWT images ===
    classes = np.unique(y)
    for class_label in classes:
        class_folder = os.path.join(output_path, str(class_label))
        os.makedirs(class_folder, exist_ok=True)

        class_data = X[y == class_label]
        for index, signal in enumerate(class_data):
            signal = signal.numpy()
            coefficients, _ = pywt.cwt(signal, scales, wavelet)

            plt.imshow(coefficients, cmap='jet', aspect='auto')
            plt.axis('off')
            image_path = os.path.join(class_folder, f'cwt_image_{class_label}_{index}.png')
            print(f"Saving: {image_path}")
            plt.savefig(image_path, transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close()


if __name__ == "__main__":
    # === Configuration ===
    file_path = r'A1_footstep_feat_full.mat'  # Path to your input .mat file
    output_path = r'C:\Users\mainak\Desktop\pythonProject\A1_new'  # Where to save images
    footsteps_num = 1  # Change if you want to aggregate more footsteps per sample

    # === Run the full processing pipeline ===
    generate_cwt_images(file_path, output_path, footsteps_num)
