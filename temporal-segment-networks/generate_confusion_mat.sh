# generate confusion matrix, and dump both the matrix file and the image.

output_folder=results/confusion_mat
weights_config='--score_weights 1 1.5'
split=1

for data in ucf101 hmdb51
do
    python tools/generate_confusion_matrix.py results/scores/score_${data}_rgb_${split}.npz results/scores/score_${data}_flow_${split}.npz results/confusion_mat/${data}_confusion.dat results/confusion_mat/${data}_confusion.png ${weights_config}
done
echo 'finished.'

