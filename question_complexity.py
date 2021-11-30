import numpy as np 

squad_question_lengths = np.loadtxt('./overlap_analysis/squad_eval_set_question_lengths.txt')
print(np.mean(squad_question_lengths))

aqa_question_lengths = np.loadtxt('./overlap_analysis/aqa_eval_set_question_lengths.txt')
print(np.mean(aqa_question_lengths))