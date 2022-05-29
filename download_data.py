import os
from urllib.request import urlopen
from shutil import copyfileobj
from bz2 import BZ2File
import gdown


drive_ids = {
    "stylegan2-ffhq-config-f.pt": "1Yr7KuD959btpmcKGAUsbAk5rPjX2MytK",
    "e4e_ffhq_encode.pt": "1o6ijA3PkcewZvwJJ73dJ0fxhndn0nnh7",
    "restyle_psp_ffhq_encode.pt": "1nbxCIVw9H3YnQsoIPykNEFwWJnHVHlVd",
    "arcane_caitlyn.pt": "1gOsDTiTPcENiFOrhmkkxJcTURykW1dRc",
    "arcane_caitlyn_preserve_color.pt": "1cUTyjU-q98P75a8THCaO545RTwpVV-aH",
    "arcane_jinx_preserve_color.pt": "1jElwHxaYPod5Itdy18izJk49K1nl4ney",
    "arcane_jinx.pt": "1quQ8vPjYpUiXM4k1_KIwP4EccOefPpG_",
    "arcane_multi_preserve_color.pt": "1enJgrC08NpWpx2XGBmLt1laimjpGCyfl",
    "arcane_multi.pt": "15V9s09sgaw-zhKp116VHigf5FowAy43f",
    "sketch_multi.pt": "1GdaeHGBGjBAFsWipTL0y-ssUiAqk8AxD",
    "disney.pt": "1zbE2upakFUAx8ximYnLofFwfT8MilqJA",
    "disney_preserve_color.pt": "1Bnh02DjfvN_Wm8c4JdOiNV4q9J7Z_tsi",
    "jojo.pt": "13cR2xjIBj8Ga5jMO7gtxzIJj2PDsBYK4",
    "jojo_preserve_color.pt": "1ZRwYLRytCEKi__eT2Zxv1IlV6BGVQ_K2",
    "jojo_yasuho.pt": "1grZT3Gz1DLzFoJchAmoj3LoM9ew9ROX_",
    "jojo_yasuho_preserve_color.pt": "1SKBu1h0iRNyeKBnya_3BBmLr4pkPeg_L",
    "art.pt": "1a0QDEHwXQ6hE_FcYEyNMuv5r5UnRQLKT",
}


def download_face_landmarks_model():
    '''
    Downloads the shape predictor model to detect facial landmarks
    which has various use cases such as aligning the face.
    '''

    face_landmarks_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    file_name = "shape_predictor_68_face_landmarks.dat.bz2"
    model_name = "dlibshape_predictor_68_face_landmarks.dat"
    with urlopen(face_landmarks_url) as f_in, open(file_name, "wb") as f_out:
        copyfileobj(f_in, f_out)
    data = BZ2File(file_name).read() 
    with open(file_name[:-4], "wb") as f:
        f.write(data)
    os.rename(file_name[:-4], os.path.join("models", model_name))
    os.remove(file_name)


def download_from_drive(file_name):
    '''Downloads any file from a Google Drive link's ID.'''
    
    file_dst = os.path.join("models", file_name)
    if not os.path.exists(file_dst):
        gdown.download(id=drive_ids[file_name], output=file_dst)


def main():
    '''Downloads the facial landmarks, pretrained StyleGAN2 and e4e model.'''

    download_face_landmarks_model()
    download_from_drive("stylegan2-ffhq-config-f.pt")
    download_from_drive("e4e_ffhq_encode.pt")
    

if __name__ == "__main__":
    main()