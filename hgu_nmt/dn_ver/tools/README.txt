Command for vocabulary set generation

- Vocabulary set for a single dataset
$ sudo python3.8 preprocess.py input_file --vocab=[vocab size]

- Vocabulary set for multiple datasets : Joined dictionary
$ sudo python3.8 preprocess.py input1 input2 ... --vocab=[vocab size]
