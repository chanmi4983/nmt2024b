GPU_ID=$1
TASK=$2 # iwslt14_ende, wmt14_ende, wmt16_enro, wmt18_enfi
MODEL=$3


if [ $TASK == 'iwslt14_ende' ]
then
    # IWSLT14 ENDE
    bash test_iwslt14_ende.sh $GPU_ID 1
elif [ $TASK == 'wmt14_ende' ]
then
    echo "Note that newstest14 has different testsets for en2de, de2en"
    # WMT14 ENDE
    bash test_wmt14_ende.sh $GPU_ID 1 en de 2013

    bash test_wmt14_ende.sh $GPU_ID 1 en de 2014
    bash test_wmt14_ende.sh $GPU_ID 1 de en 2014

    bash test_wmt14_ende.sh $GPU_ID 1 en de 2015
else
    echo "Wrong task name"
fi
