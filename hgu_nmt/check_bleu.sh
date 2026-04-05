#FILE='kr2en.gpu6x1.0()+x(w**0.5).bleu'
#python3 check_bleu.py --file=$FILE

#FILE='kr2en.gpu5scale1.()+x.temp.no0.5.bleu'
#python3 check_bleu.py --file=$FILE

#FILE='kr2en.gpu4scale1.()+x.temp.no0.5.trainable.bleu'
#python3 check_bleu.py --file=$FILE
#FILE='kr2en.gpu4scale1.()+x.temp.no0.5.trainable.best.pth'
#python3 check_model.py --file=$FILE

#FILE='kr2en.gpu7scale1.()+x.temp.no0.5.max.bleu'
#python3 check_bleu.py --file=$FILE


#FILE='kr2en.gpu3pre_norm.no_drop.bleu'
#python3 check_bleu.py --file=$FILE
#FILE='kr2en.gpu4scale1.()+x.temp.no0.5.trainable.best.pth'
#python3 check_model.py --file=$FILE

#FILE='kr2en.gpu6scale1.()+x.temp.no0.5.0inhibitory.bleu'
#python3 check_bleu.py --file=$FILE

#FILE='kr2en.gpu7pre_norm.train0.inhibit1.train_eval.bleu'
#python3 check_bleu.py --file=$FILE

#FILE='kr2en.gpu4pre_norm.train0.inhibit1.train_eval.bleu'
#python3 check_bleu.py --file=$FILE

#FILE='kr2en.gpu5scale1.()+x.temp.no0.5.train.bleu'
#python3 check_bleu.py --file=$FILE

#FILE='kr2en.gpu2pre_norm.bern0.8_train.bleu'
#python3 check_bleu.py --file=$FILE


FILE='kr2en.gpu4_dk32.dv64.bleu'
python3 check_bleu.py --file=$FILE

FILE='kr2en.gpu1_dk64.self.cross.bleu'
python3 check_bleu.py --file=$FILE
