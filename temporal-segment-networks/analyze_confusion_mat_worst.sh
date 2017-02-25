k=5

echo '======================================='
echo 'analyzing confusion matrix on UCF101...'
python tools/analyze_confusion_matrix.py --dump_file results/confusion_mat/ucf101_confusion_analysis.dat  results/confusion_mat/ucf101_confusion.dat data/ucf101_splits/classInd.txt ${k}
echo '======================================='
echo 'analyzing confusion matrix on HMDB51...'
python tools/analyze_confusion_matrix.py --dump_file results/confusion_mat/hmdb51_confusion_analysis.dat results/confusion_mat/hmdb51_confusion.dat data/hmdb51_splits/class_list.txt ${k}

echo 'Done.'
