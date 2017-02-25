echo 'fetching sample failure cases from generated confusion matrix analysis'

num_sample=1
output_dir=results/failure_cases

python tools/grab_failure_cases.py results/scores/score_ucf101_rgb_1.npz results/scores/score_ucf101_flow_1.npz results/confusion_mat/ucf101_confusion_analysis.dat ${num_sample} ${output_dir}/ucf101.dat ../../data/ucf101_frame_and_flow ucf101  --score_weights 1 1.5

python tools/grab_failure_cases.py results/scores/score_hmdb51_rgb_1.npz results/scores/score_hmdb51_flow_1.npz results/confusion_mat/hmdb51_confusion_analysis.dat ${num_sample} ${output_dir}/hmdb51.dat ../../data/hmdb51_frame_and_flow hmdb51  --score_weights 1 1.5
