# sorting along the genome
from w2v_lstm_functions import run_w2v

drug = 'Pyrazinamide'
fld = '1'
for fld in ['2', '3', '4', '5']:
    run_w2v(drug, fld)