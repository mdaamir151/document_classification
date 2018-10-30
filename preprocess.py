
import _preprocess
import config

def run(arg,gram_sz=3,split_size=10):
	if arg == False:
		return
	preprocessor = _preprocess.preprocess(config.language_dir,gram_sz)
	preprocessor._toLower(config.language_dir,config.lower_dir)._stem(config.lower_dir,config.stem_dir)._makeNgrams(config.stem_dir,config.grams_dir)._stats(config.grams_dir,config.stats_dir)._splitFiles(config.grams_dir,config.train_dir,split_size)
