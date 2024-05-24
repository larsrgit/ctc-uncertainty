Download and install the correct speechbrain version
    
    bash
    git clone https://github.com/speechbrain/speechbrain.git
    cd speechbrain
    git checkout 0110f4e65e2c866e18d49c1a82c7ef00fe4959ca
    pip install -r requirements.txt
    pip install --editable .

Install other requirements:
    
    pandas
    matplotlib
    kaldialign

Download checkpoints. The links are provided in the READMEs from the corresponding recipes of speechbrain:

mcv: https://drive.google.com/drive/folders/19G2Zm8896QSVDqVfs7PS_W86-K0-5xeC
(provided by https://github.com/speechbrain/speechbrain/tree/0110f4e65e2c866e18d49c1a82c7ef00fe4959ca/recipes/CommonVoice/ASR/CTC)



timit: https://drive.google.com/drive/folders/1OhBOTfC34PaOuiLIUjEBP1JmmlBTxJ8D
(provided by https://github.com/speechbrain/speechbrain/tree/0110f4e65e2c866e18d49c1a82c7ef00fe4959ca/recipes/TIMIT/ASR/CTC)

Download TIMIT and Mozilla Common Voice (paper uses MCV Version cv-corpus-11.0-2022-09-21/de)


change following parameters in both hyperparams_DATASET.yaml to the corresponding paths where you saved the checkpoints,
or save the checkpoints according to the parameters set:
    
    output_folder
    wer_file
    save_folder
    train_log

change the data paths (strg+F "/path/to") to where you saved the datasets in both hyperparams_DATASET.yaml

run
```
python uncertainty_infer_common_voice.py hyperparams_mcv.yaml
python uncertainty_infer_timit.py hyperparams_timit.yaml
```