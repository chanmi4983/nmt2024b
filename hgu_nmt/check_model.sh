#FILE='kr2en.gpu6.lr0.0001.label0.01.pth' # trainable temper
#FILE='kr2en.gpu7.lr0.0001.label0.01.pth' # non trainable temper
#FILE='kr2en.gpu5.lr0.0001.label0.01.pth'

#FILE='kr2en.gpu6x0.7x0.5(+x).best.pth'
#python3 check_model.py --file=$FILE

#FILE='kr2en.gpu7x1.4x1.0()+x.best.pth'
#python3 check_model.py --file=$FILE

#FILE='kr2en.gpu4scale1.()+x.temp.no0.5.trainable.pth'
#FILE='kr2en.gpu5scale1.()+x.temp.no0.5.train.pth'
FILE='kr2en.gpu6.org.scale_para2.pth'
python3 check_model.py --file=$FILE

#FILE='kr2en.gpu4scale1.()+x.temp.no0.5.train.inhibit.pth'
FILE='kr2en.gpu7.org.ed_ln_scale_para.pth'
python3 check_model.py --file=$FILE
