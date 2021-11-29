from datasets import load_dataset, concatenate_datasets

squad = load_dataset('squad', split='validation')
aqa = load_dataset('adversarial_qa', 'adversarialQA', split='validation')

aqa = aqa.remove_columns("metadata")

assert squad.features.type == aqa.features.type
squad_aqa_dataset = concatenate_datasets([squad, aqa])
print(squad_aqa_dataset.shape)

squad_aqa_dataset.save_to_disk('data/eval/')