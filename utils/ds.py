from modelscope.msdatasets import MsDataset

dataset_id = "Muennighoff/natural-instructions"
for split in ["train", "validation", "test"]:
    MsDataset.load(dataset_id, split=split)
print("done")