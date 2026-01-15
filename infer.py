import os
import torch
import soundfile as sf
from train.cgtfcrn import CGTFCRN

CGTFCMCnt = 4
## load model
device = torch.device("cpu")
model = CGTFCRN(convCnt=CGTFCMCnt).eval()
total_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of parameters:{total_params}")
#torch.save(model,"checkpoints/new.pth")
if CGTFCMCnt == 4:
    ckpt = torch.load(os.path.join('checkpoints', 'model_0120.tar'), map_location=device)
elif CGTFCMCnt == 3:
    ckpt = torch.load(os.path.join('checkpoints', 'model_0120_3.tar'), map_location=device)
model.load_state_dict(ckpt['model'])

test_file_list=[
'Chips_Near_Regular_SP_Mobile_Primary_noisy',
'Street-Creaky_Far_Regular_SP_Desk_Primary_noisy',
'Water_Near_Sad_SP_Desk_Primary_noisy',
'winding_workshop_noisy',
]
for test_file in test_file_list:
    ## load data
    mix, fs = sf.read(os.path.join('test_wavs', test_file + ".wav"), dtype='float32')
    assert fs == 16000

    ## inference
    input = torch.stft(torch.from_numpy(mix), 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)
    with torch.no_grad():
        output = model(input[None])[0]
    enh = torch.istft(output, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)

    ## save enhanced wav
    sf.write(os.path.join('test_wavs', test_file + '_out.wav'), enh.detach().cpu().numpy(), fs)
