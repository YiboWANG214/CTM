# Reference
https://github.com/yinwenpeng/BenchmarkingZeroShot

https://github.com/helboukkouri/character-bert

https://github.com/mkshing/Prompt-Tuning

# Requirements

For entity typing / textual entailment phase:

```
pip install transformers==3.3.1 scikit-learn==0.23.2
```

For prompt tuning phase:

```
pip install transformers==4.8.2

pip install jsonlines
```



# Usage

For BERT+entailment:

```
cd src
%run Entailment.py  --task_name rte --do_train --do_lower_case --max_seq_length 64 --train_batch_size 16 --learning_rate 3e-6 --num_train_epochs 10  --data_dir '' --output_dir ''
```

For BERT entity typing:

```
cd src
%run BERT.py  --task_name rte --do_train --do_lower_case --max_seq_length 64 --train_batch_size 32 --learning_rate 5e-6 --num_train_epochs 10  --data_dir '' --output_dir ''
```

For CTM Models:

```
cd src
cd character-bert
python prompt_fusion.py  --task_name rte --do_train --do_lower_case --max_seq_length 64 --train_batch_size 16 --learning_rate 5e-6 --num_train_epochs 10  --data_dir '' --output_dir ''
```

For CTM w/o Prompt Tuning:

```
cd src
cd character-bert
python fusion.py  --task_name rte --do_train --do_lower_case --max_seq_length 64 --train_batch_size 16 --learning_rate 1e-5 --num_train_epochs 10  --data_dir '' --output_dir ''
```

For CTM w/o Fusion Embedding:

```
cd src
cd character-bert
python prompt_fusion_BERT.py  --task_name rte --do_train --do_lower_case --max_seq_length 64 --train_batch_size 16 --learning_rate 5e-6 --num_train_epochs 10  --data_dir '' --output_dir ''
```

or

```
python prompt_fusion_CharacterBERT.py  --task_name rte --do_train --do_lower_case --max_seq_length 64 --train_batch_size 16 --learning_rate 5e-6 --num_train_epochs 10  --data_dir '' --output_dir ''
```

For Prompt Tuning Phase:

```
# for BERT
python prompt_tuning_train.py
# for CharacterBERT
python character_prompt_tuning_train.py
```



Notes: 

Change type2hypothesis for different hypotheses

fusion methods can be changed