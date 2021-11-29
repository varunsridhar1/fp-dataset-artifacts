from datasets import load_dataset
import numpy as np
import re

squad = load_dataset('squad', split='validation')
aqa = load_dataset('adversarial_qa', 'adversarialQA', split='validation')

squad_question_lengths = np.zeros((squad.shape[0],))
squad_overlap = np.zeros((squad.shape[0],))
aqa_question_lengths = np.zeros((aqa.shape[0],))
aqa_overlap = np.zeros((aqa.shape[0],))

for i in range(len(squad)):
    squad_question_lengths[i] = len(list(set(re.split('\W+', squad[i]['question']))))
    squad_overlap[i] = len(list(set(re.split('\W+', squad[i]['question'])) & set(re.split('\W+', squad[i]['context']))))

for i in range(len(aqa)):
    aqa_question_lengths[i] = len(list(set(re.split('\W+', aqa[i]['question']))))
    aqa_overlap[i] = len(list(set(re.split('\W+', aqa[i]['question'])) & set(re.split('\W+', aqa[i]['context']))))

np.savetxt('./overlap_analysis/squad_eval_set_question_lengths.txt', squad_question_lengths)
np.savetxt('./overlap_analysis/squad_eval_set_overlap.txt', squad_overlap)
np.savetxt('./overlap_analysis/aqa_eval_set_question_lengths.txt', aqa_question_lengths)
np.savetxt('./overlap_analysis/aqa_eval_set_overlap.txt', aqa_overlap)