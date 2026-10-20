"""Calculate ACC/UAR/UF1 from real exported predictions; never fabricate logits."""
import argparse,csv,json
from pathlib import Path
from sklearn.metrics import accuracy_score,recall_score,f1_score,confusion_matrix
from protocol import NUM_CLASSES,NAME

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--predictions',type=Path,required=True,help='CSV with true/pred labels')
 p.add_argument('--label-column',default='Label')
 p.add_argument('--prediction-column',default='ReplayPrediction')
 p.add_argument('--output',type=Path)
 a=p.parse_args()
 with a.predictions.open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
 if not rows:p.error('Empty predictions CSV')
 try:
  yt=[int(r[a.label_column]) for r in rows]
  yp=[int(r[a.prediction_column]) for r in rows]
 except (KeyError,ValueError) as e:p.error(f'Invalid label/prediction columns: {e}')
 labels=list(range(NUM_CLASSES))
 if any(x not in labels for x in yt+yp):p.error('Labels are not in the 0..2 Combined3 label space')
 result={'protocol':NAME,'samples':len(rows),'ACC':float(accuracy_score(yt,yp)),
 'UAR':float(recall_score(yt,yp,labels=labels,average='macro',zero_division=0)),
 'UF1':float(f1_score(yt,yp,labels=labels,average='macro',zero_division=0)),
 'confusion_matrix':confusion_matrix(yt,yp,labels=labels).tolist(),
 'metric_note':'macro classes [0,1,2]; missing classes score zero'}
 out=json.dumps(result,indent=2)
 print(out)
 if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(out+'\n',encoding='utf-8')
if __name__=='__main__':main()
