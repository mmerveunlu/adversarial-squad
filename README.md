
This repo was forked from [Adversarial SQuAD](https://github.com/robinjia/adversarial-squad)

Original README file can be found as README-orig.md

Several changes, I made:
 * `print` functions were changed for Python3 as `print`().
 * In convert_questions.py, there was an error, because `patten` written instead of pattern.
 * `unicode()` functions were either removed or replaced with `str()`.
 * An argument for output file was added.
 * SQuAD words were removed.

### Before Running:
 * Check Java version, it should be 8. I have followed [the answer on this link](https://askubuntu.com/questions/1133216/downgrading-java-11-to-java-8) to downgrade
 * Check the requirements, especially [pattern](https://github.com/clips/pattern#installation). Python3 repo of [pattern3 is not maintained anymore](https://github.com/clips/pattern/issues/62#issuecomment-370766376), so check the github repo to download it.
 * Check the StandfordCoreNLP by running separately. I followed [this link](https://www.khalidalnajjar.com/setup-use-stanford-corenlp-server-python/).
 * Run the `pull-dependencies.sh` which will download SQuAD and Glove.

### Intructions to Run: 

 * Under the main project folder, create an output folder.
   `>> mkdir out-exp`
 * Precompute nearby words from given file
   `>> python src/py/find_squad_nearby_words.py glove/glove.6B.100d.txt -n 100 -f data/squad/train-v1.1.json -o out-exp/nearby_n100_glove_6B_100d.json`
 * Check the folder paths in convert_questions.py   
 * Run CoreNLP
   `>> python src/py/convert_questions.py corenlp -d train`
 * Create Examples
   `>> python src/py/convert_questions.py dump-highConf -d train -q`

### ToDo
 * Run the examples with BERT, MatchLSTM, Bidaf and Mreader
 * Change the parser in `convert_questions()`, take the dataset/folder paths and make them arguments

