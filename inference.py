# import argparse
# import os
# from os.path import abspath, dirname, join
# from imageio.v3 import imread
# import numpy as np
# import sofar
# import torch


# AVG_HRTF_PATH = join(dirname(abspath(__file__)), 'data', 'Average_HRTFs.sofa')


# class BaselineHRTFPredictor:
#     def __init__(self, average_hrtf_path: str = AVG_HRTF_PATH):
#         """Creates a predictor instance. average_HRTF_path is the path the file 'Average_HRTFs.sofa' that was delivered as part of the project."""
#         self.average_hrir = sofar.read_sofa(average_hrtf_path, verbose=False)

#     def predict(self, images: torch.Tensor) -> sofar.Sofa:
#         """
#         Predict the HRTF based on left and right images.

#         Args:
#             images: images for left and right pinna as 4-dimensional tensor of size (number of ears, number of images per ear, image height, image width)

#         Returns:
#             sofar.Sofa: Predicted HRIR in SOFA format.
#         """
#         return self.average_hrir


# def main():
#     parser = argparse.ArgumentParser(description="Baseline HRTF Inference Script")
#     parser.add_argument("-l", "--left", metavar='IMAGE_PATH', type=str, nargs='+', required=True, help="List of left pinna images")
#     parser.add_argument("-r", "--right", metavar='IMAGE_PATH', type=str, nargs='+', required=True, help="List of right pinna images")
#     parser.add_argument("-o", "--output_path", metavar='SOFA_PATH', type=str, required=True, help="File path to save the predicted HRTF in SOFA format.")
#     args = parser.parse_args()

#     # load images
#     left_images = torch.from_numpy(np.stack([imread(path) for path in args.left]))
#     right_images = torch.from_numpy(np.stack([imread(path) for path in args.right]))
#     images = torch.stack((left_images, right_images))

#     # predict HRTFs
#     predictor = BaselineHRTFPredictor()
#     hrtf = predictor.predict(images)

#     # write to output path
#     os.makedirs(dirname(args.output_path), exist_ok=True)
#     sofar.write_sofa(args.output_path, hrtf, compression=0)
#     print(f"Saved HRTF to {args.output_path}")

# if __name__ == "__main__":
#     main()


import argparse
import os
from os.path import dirname
import numpy as np
import torch
from imageio.v3 import imread
import sofar
from model import HRTFGenerator  # Import our model

class HRTFPredictor:
    def __init__(self, model_path='best_model.pth', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = HRTFGenerator().to(device)
        
        # Charger le modèle avec des clés filtrées
        checkpoint = torch.load(model_path, map_location=device)
        filtered_state_dict = {
            k: v for k, v in checkpoint.items() if k in self.model.state_dict()
        }
        self.model.load_state_dict(filtered_state_dict, strict=False)  # Ignorer les incompatibilités
        self.model.eval()

    def predict(self, images: torch.Tensor) -> sofar.Sofa:
        """
        Predict the HRTF based on left and right images.
        
        Args:
            images: images for left and right pinna as 4-dimensional tensor of shape 
                   (number of ears, number of images per ear, image height, image width)
        
        Returns:
            sofar.Sofa: Predicted HRIR in SOFA format
        """
        with torch.no_grad():
            images = images.to(self.device).float() / 255.0  # Normalize images
            hrtfs = self.model(images.unsqueeze(0))  # Add batch dimension
        
            # Convert to HRIR
            hrirs = np.fft.irfft(hrtfs[0].cpu().numpy(), axis=-1)
            
            # Create SOFA object (you'll need to set appropriate metadata)
            sofa_obj = sofar.Sofa(convention="SimpleFreeFieldHRIR")# Ajout de l'argument convention
            sofa_obj.Data_IR = hrirs
            sofa_obj.Data_SamplingRate = 44100  # Exemple
            sofa_obj.ListenerPosition = [0, 0, 0]  # Exemple de métadonnées

            
            return sofa_obj

def main():
    parser = argparse.ArgumentParser(description="HRTF Inference Script")
    parser.add_argument("-l", "--left", metavar='IMAGE_PATH', type=str, nargs='+', 
                       required=True, help="List of left pinna images")
    parser.add_argument("-r", "--right", metavar='IMAGE_PATH', type=str, nargs='+', 
                       required=True, help="List of right pinna images")
    parser.add_argument("-o", "--output_path", metavar='SOFA_PATH', type=str, 
                       required=True, help="File path to save the predicted HRTF in SOFA format.")
    args = parser.parse_args()

    # Load images
    left_images = torch.from_numpy(np.stack([imread(path) for path in args.left]))
    right_images = torch.from_numpy(np.stack([imread(path) for path in args.right]))
    images = torch.stack((left_images, right_images))

    # Predict HRTFs
    predictor = HRTFPredictor()
    hrtf = predictor.predict(images)

    # Write to output path
    os.makedirs(dirname(args.output_path), exist_ok=True)
    sofar.write_sofa(args.output_path, hrtf, compression=0)
    print(f"Saved HRTF to {args.output_path}")

if __name__ == "__main__":
    main()