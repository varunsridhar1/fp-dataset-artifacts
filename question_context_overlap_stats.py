import numpy as np

squad_question_lengths = np.loadtxt('./overlap_analysis/squad_eval_set_question_lengths.txt')
squad_overlap = np.loadtxt('./overlap_analysis/squad_eval_set_overlap.txt')

# mean percentage of overlap
squad_percent_question_overlap = squad_overlap / squad_question_lengths
print(np.mean(squad_percent_question_overlap) * 100)

aqa_question_lengths = np.loadtxt('./overlap_analysis/aqa_eval_set_question_lengths.txt')
aqa_overlap = np.loadtxt('./overlap_analysis/aqa_eval_set_overlap.txt')

aqa_percent_question_overlap = aqa_overlap / aqa_question_lengths
print(np.mean(aqa_percent_question_overlap) * 100)

