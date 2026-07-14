from datasets import load_dataset
import os

os.environ['HF_HOME']='/opt/local/data/'


ds = load_dataset("OpenRLHF/preference_dataset_mixture2_and_safe_pku")

ds.save_to_disk("/opt/nas/p/mmu/zb/DATA/raw_data/hf_data/OpenRLHF/preference_dataset_mixture2_and_safe_pku")
