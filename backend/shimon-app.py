from scipy.io.wavfile import write
import whisper
import os
import torch
import subprocess
from nemo.collections.tts.models import FastPitchModel, HifiGanModel
import numpy as np
import time
from datetime import datetime
import pytz
from pydub import AudioSegment
import json
from agent import rap_battle, end_session
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
from ns3_codec import FACodecDecoderV2, FACodecEncoderV2
from huggingface_hub import hf_hub_download
import librosa
import soundfile as sf
import pickle
import time
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Location", "X-Rap-Transcript"],
)

# Audio settings
sampleRate = 22050
inputDir = "recordings"
outputDir = "synthesized"
transcriptFile = "transcript.txt"
tempo = 100
beatInterval = 60 / tempo

os.makedirs(inputDir, exist_ok=True)
os.makedirs(outputDir, exist_ok=True)

# Text-to-speech settings
desired_gpu = 1
if torch.cuda.is_available():
    torch.cuda.set_device(desired_gpu)
    device = torch.device(f"cuda:{desired_gpu}")
    print(f"Using GPU {desired_gpu}: {torch.cuda.get_device_name(desired_gpu)}")
else:
    device = torch.device("cpu")
    print("CUDA not available, using CPU")

#vocoder = HifiGanModel.from_pretrained("tts_en_hifigan").eval().to(device)
checkpoint_path = "/home/dat/rap_synth/ljspeech_to_9017_no_mixing_5_mins/FastPitch/2025-09-01_16-47-10/checkpoints/FastPitch--val_loss=1.0077-epoch=9-last.ckpt"
spec_model = FastPitchModel.load_from_checkpoint(
    checkpoint_path,
    map_location=lambda storage, loc: storage.cuda(desired_gpu)
).eval().to(device)

#vocoder_path = "/home/dat/rap_synth/hifigan_ft/HifiGan/2025-08-14_10-54-40/checkpoints/HifiGan--val_loss=0.9787-epoch=119-last.ckpt"

#vocoder = HifiGanModel.load_from_checkpoint(checkpoint_path=vocoder_path, map_location=lambda storage, loc: storage.cuda(desired_gpu)).eval().to(device)

vocoder = HifiGanModel.from_pretrained("tts_en_hifigan")
vocoder = vocoder.eval().cuda()

whisper_model = whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")

with open("speaker_embs.pkl", "rb") as f:
    speaker_embs = pickle.load(f)

try:
    # Start Ollama server in background
    #subprocess.run(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Preload phi4 model
    ollama_process = subprocess.Popen(
        ["ollama", "run", "phi4"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
    )
    print("Started Ollama with phi4 model")
    # Wait briefly to ensure model is loaded
    time.sleep(2)
except Exception as e:
    print(f"Error starting Ollama: {e}")
    ollama_process = None


fa_encoder_v2 = FACodecEncoderV2(
    ngf=32,
    up_ratios=[2, 4, 5, 5],
    out_channels=256,
)

fa_decoder_v2 = FACodecDecoderV2(
    in_channels=256,
    upsample_initial_channel=1024,
    ngf=32,
    up_ratios=[5, 5, 4, 2],
    vq_num_q_c=2,
    vq_num_q_p=1,
    vq_num_q_r=3,
    vq_dim=256,
    codebook_dim=8,
    codebook_size_prosody=10,
    codebook_size_content=10,
    codebook_size_residual=10,
    use_gr_x_timbre=True,
    use_gr_residual_f0=True,
    use_gr_residual_phone=True,
)

encoder_v2_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder_v2.bin")
decoder_v2_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder_v2.bin")

fa_encoder_v2.load_state_dict(torch.load(encoder_v2_ckpt))
fa_decoder_v2.load_state_dict(torch.load(decoder_v2_ckpt))

fa_encoder_v2 = fa_encoder_v2.to(device).eval()
fa_decoder_v2 = fa_decoder_v2.to(device).eval()


file_path = '/home/dat/rap_synth/beats/beat_times.pkl'
with open(file_path, 'rb') as file:
    global_beat_data = pickle.load(file)

#helper functions for voice conversion

def load_audio(wav_input, target_sr = 16000):
    if isinstance(wav_input, str):
        # It's a file path
        wav = librosa.load(wav_input, sr=target_sr)[0]
        wav = torch.from_numpy(wav).float()
    elif isinstance(wav_input, np.ndarray):
        if wav_input.dtype == np.int16:
            wav = wav_input.astype(np.float32) / 32767.0
        else:
            wav = wav_input
        
        wav = librosa.resample(wav, orig_sr=22050, target_sr=target_sr)
        wav = torch.from_numpy(wav).float()
    else:
        raise TypeError(f"Expected str or np.ndarray, got {type(wav_input)}")
    
    downsample_factor = 200
    current_length = wav.shape[0]
    if current_length % downsample_factor != 0:
        pad_length = downsample_factor - (current_length % downsample_factor)
        wav = torch.nn.functional.pad(wav, (0, pad_length), mode='constant', value=0)
    
    wav = wav.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, T) and move to GPU
    return wav



#returns first speaker emb
def get_speaker_embs(wav_path):
    wav = load_audio(wav_path)
    with torch.no_grad():
        enc_out = fa_encoder_v2(wav)
        prosody = fa_encoder_v2.get_prosody_feature(wav)
        vq_post_emb, vq_id, _, quantized, spk_embs = fa_decoder_v2(enc_out, prosody, eval_vq=False, vq=True)
        return spk_embs

def voice_conversion(input_wav_path, output_wav_path, output_spk_embs, blend_rate = 0.5):
    wav = load_audio(input_wav_path)
    with torch.no_grad():
    
        enc_out = fa_encoder_v2(wav)
        prosody = fa_encoder_v2.get_prosody_feature(wav)
        vq_post_emb, vq_id, _, quantized, input_spk_embs = fa_decoder_v2(
        enc_out, prosody, eval_vq=False, vq=True
        )

        # Ensure speaker embeddings are on the same device before blending
        output_spk_embs = output_spk_embs.to(input_spk_embs.device)
        blended_spk_embs = (1 - blend_rate) * input_spk_embs + blend_rate * output_spk_embs

        vq_post_emb_converted = fa_decoder_v2.vq2emb(vq_id, use_residual=False)
        recon_wav_converted = fa_decoder_v2.inference(vq_post_emb_converted, blended_spk_embs)
        
        
    # Move to CPU before converting to numpy
    sf.write(output_wav_path, recon_wav_converted[0][0].detach().cpu().numpy(), 16000)

#input_spk_embs: list[np.ndarray(1,)])
def meld_embs(input_spk_embs, proportions):
    output_spk_emb = np.zeros((256,))
    for i in range(0, len(input_spk_embs)):
        weighted_emb = proportions[i] * input_spk_embs[i]
        output_spk_emb += weighted_emb
        
    # Return on the global model device to avoid CPU/GPU mixing
    return torch.tensor(np.array([output_spk_emb]), dtype=torch.float, device=device)





@app.post("/generate")
async def rap_generation(
    recording: UploadFile = File(...),
    tones: str = Form(...),
    voices: str = Form(...),
    thread_id: str = Form(...),
    turn: int = Form(...)):

    recording_file = os.path.join(inputDir, f"session_{thread_id}_human_{turn}.wav")
    try:
        with open(recording_file, 'wb') as f:
            f.write(await recording.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {e}")

    try:
        tones = json.loads(tones)
        voices = json.loads(voices)

        print(f"tones: {tones}")
        print(f"voices: {voices}")
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON for parameters. {e}")

    output_path = os.path.join(outputDir, f"session_{thread_id}_robot_{turn}.wav")
    _, output_path, rap_transcript = process_recording(recording_file, output_path, tones, voices, thread_id)


    response = FileResponse(
        output_path,
        media_type="audio/wav",
        filename=f"session_{thread_id}_human_{turn}.wav"
    )

    print(rap_transcript)
    response.headers["X-Rap-Transcript"] = base64.b64encode(rap_transcript.encode("utf-8")).decode("ascii")
    return response

@app.post("/end")
async def end_session(
    thread_id: str = Form(...)
):
    try:
        end_session(thread_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{e}")


def transcribeAudio(filePath: str, savePath: str):
    print("Transcribing...")
    result = whisper_model.transcribe(filePath)
    with open(savePath, "w") as file:
        file.write(result["text"])
    return result["text"]

def infer(spec_gen_model, vocoder_model, str_input, speaker=None):
    with torch.no_grad():
        parsed = spec_gen_model.parse(str_input)
        if speaker is not None:
            speaker = torch.tensor([speaker]).long().to(device=spec_gen_model.device)
        spectrogram = spec_gen_model.generate_spectrogram(tokens=parsed, speaker=speaker, pace=0.8) #add pace here
        audio = vocoder_model.convert_spectrogram_to_audio(spec=spectrogram)
    if isinstance(audio, torch.Tensor):
        audio = audio.to('cpu').numpy()
    if audio.ndim == 2:
        audio = audio[0]
    return audio

def generateRap(transcript, emotion_mix, thread_id):
    try:
        #prompt = f"You are a diss battle rapper. Respond to the following insult using only a 8 line bar that rhymes, and output only those 8 lines: {transcript}"
        #env = os.environ.copy()
        #env["CUDA_VISIBLE_DEVICES"] = "0"
        #result = subprocess.run(
        #    ['ollama', 'run', 'phi4', prompt],
        #    text=True,
        #    capture_output=True,
        #    encoding="utf-8",
        #    check=True,
        #    env=env
        #)
        result = rap_battle(transcript, emotion_mix, ["related to human given context"], thread_id)
        return result
    except Exception as e:
        print(f"Error: {e}")
        return "Error: Could not generate response."

def process_recording(recording_file, output_path, emotion_mix, voice_mix, thread_id):
    transcription = transcribeAudio(recording_file, transcriptFile)
    print("Transcription:", transcription)
    if transcription.lower() in ["stop", "exit", "quit"]:
        return False, None
    
    response = generateRap(transcription, emotion_mix, thread_id)
    print("Generated Response:", response)
    
    audio = infer(spec_model, vocoder, response)
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Save the audio directly without metronome
    write(output_path, sampleRate, audio_int16)
    print(f"Synthesized rap saved as {output_path}")

    original_path = output_path.replace(".wav", "_original.wav")
    write(original_path, sampleRate, audio_int16)
    print(f"Original TTS audio saved as {original_path}")

    speakers = [speaker_embs[i] for i in voice_mix.keys() if i != "default"]
    rates = [j for i, j  in voice_mix.items()]


    converted_path = output_path.replace(".wav", "_converted.wav")
    target = meld_embs(speakers, rates)
    voice_conversion(original_path, converted_path, target, blend_rate=(1-voice_mix["default"]))
    print(f"Voice converted audio saved as {converted_path}")
    
    return True, converted_path, response