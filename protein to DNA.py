from flask import Flask, render_template, request
import torch, re
import pandas as pd 
from transformers import BertModel, BertTokenizer
import pandas as pd

import joblib

app = Flask(__name__)
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
model = BertModel.from_pretrained("Rostlab/prot_bert")


def protein_to_dna(l):
    codon_table = {
        'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
        'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
        'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
        'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',                
        'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
        'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
        'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
        'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
        'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
        'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
        'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
        'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
        'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
        'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
        'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
        'TAC':'Y', 'TAT':'Y', 'TAA':'', 'TAG':'',
        'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
    }

    
    dna_sequence = ""
    for amino_acid in l:
        for codon, aa in codon_table.items():
            if aa == amino_acid:
                dna_sequence += codon
    return dna_sequence
complementary_nucleotides = {"G": "C", "C": "G", "A": "U", "T": "A"}
def convert_to_dense_columns(features_array):    
            df = pd.DataFrame(features_array)
            df.columns = ['Feature_' + str(x) for x in df.columns]
            return df


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/getprediction', methods=['POST', 'GET'])
def getprediction():
    if request.method == 'POST':
        l = request.form['ps']
        p = request.form['ph']
        data = {'protein_sequence': l,'ph' : p}

        df1 = pd.DataFrame(data, index=[0])
        embeddings_list = []
        sequence_Example = l
        sequence_Example = re.sub(r"[UZOB]", "X", sequence_Example)
        encoded_input = tokenizer(sequence_Example, add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")
        with torch.no_grad():
            output = model(**encoded_input)
            output = output[1].detach().cpu().numpy()[0]
            embeddings_list.append(output)
        train_features =   embeddings_list
        train_feats_df = convert_to_dense_columns(train_features)
        train_feats_df["protein_length"] =  len(l)
        def return_amino_acid_df(df):    
            search_amino=['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
            for amino_acid in search_amino:
                df[amino_acid]=df['protein_sequence'].str.count(amino_acid,re.I)
            return df

        df1 = return_amino_acid_df(df1)

        # One-hot encode the 'ph' column
        df1 = pd.get_dummies(df1, columns=['ph'])

        # Drop the 'protein_sequence' column
        df1.drop(columns=["protein_sequence"], inplace=True)

        maindf = pd.concat([df1, train_feats_df], axis=1)
        pickled_model = joblib.load('model.pkl')
        Temp = pickled_model.predict(maindf)[0]
        DNA= protein_to_dna(l)
        DNA = DNA.translate(str.maketrans({'G': 'C', 'C': 'G', 'A': 'U', 'T': 'A'}))
        # Convert DNA sequence to RNA sequence
        mRNA = "".join(complementary_nucleotides.get(nt, "") for nt in DNA)
        return render_template('output.html', Temp=Temp, DNA=DNA, mRNA=mRNA)
    
       

if __name__ == "__main__":
    app.run(debug=True)