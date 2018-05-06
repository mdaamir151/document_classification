
from nltk.stem.snowball import SnowballStemmer
from nltk import ngrams
from collections import Counter
import glob
import os

"""
document files are inside indir directory
"""

class preprocess:
	def __init__(self,languageDir):
		g = glob.glob(languageDir+'/*')
		self.languages = [ l.split('/')[-1] for l in g ]
		self.num_classes = len(self.languages)
		self.gram_sz = 3 # trigram
		lm = {}
		fs = open('language_map.py','w')
		fs.write('lang_map = {\n')
		for (i,l) in enumerate(self.languages):
			fs.write("'{}' : {},\n".format(l,i))
		fs.write('}')
		fs.close()
			

	def _toLower(self,srcdir,targetdir):
		"""converts files in srcdir to lower and stores in targetdir
		"""
		assert os.path.isdir(srcdir)
		if not os.path.isdir(targetdir):	os.mkdir(targetdir)

		files = glob.glob(srcdir+'/*')
		lfiles = [os.path.join(targetdir,f.split('/')[-1]) for f in files]

		for (infile,outfile) in zip(files,lfiles) : 
			print("converting {} to lower".format(infile))
			f = open(infile,'r',encoding='ISO-8859-1')
			x = f.read()
			x = x.lower()
			fl = open(outfile,'w',encoding='ISO-8859-1')
			fl.write(x)
			f.close()
			fl.close()
		return self

	def _stem(self,srcdir,targetdir):
		"""stems files in srcdir to lower and store in targetdir
		"""
		assert os.path.isdir(srcdir)
		if not os.path.isdir(targetdir):	os.mkdir(targetdir)

		lfiles = glob.glob(srcdir+'/*')
		stemfiles = [os.path.join(targetdir,f.split('/')[-1]) for f in lfiles]

		for (lowerf,stemf,lang) in zip(lfiles,stemfiles,self.languages):
			print("stemming ",lowerf)
			stemmer = SnowballStemmer(lang)
			f = open(lowerf,'r',encoding='ISO-8859-1')
			x = f.read().split()
			y = []
			for word in x:
				w = stemmer.stem(word)
				y.append(w)
			text = " ".join(y)
			fl = open(stemf,'w',encoding='ISO-8859-1')
			fl.write(text)
			fl.close()
			f.close()
		return self

	def _makeNgrams(self,srcdir,targetdir):

		assert os.path.isdir(srcdir)
		if not os.path.isdir(targetdir):	os.mkdir(targetdir)

		stemfiles = glob.glob(srcdir+'/*')
		gramfiles = [os.path.join(targetdir,f.split('/')[-1]) for f in stemfiles]

		for (stemf,grf) in zip(stemfiles,gramfiles):
			print("Constructing {}-grams for {}".format(self.gram_sz,stemf))
			fi = open(stemf,'r',encoding='ISO-8859-1')
			x = fi.read().split()
			Ngrams = []

			for word in x:
				seq = ngrams(word,self.gram_sz)
				seq = list(seq)
				y = ["".join(ch) for ch in seq]
				Ngrams.extend(y)
			# print(Ngrams)
			ngr = " ".join(Ngrams)
			fl = open(grf,'w',encoding='ISO-8859-1')
			fl.write(ngr)
			fl.close()
			fi.close()
		return self

	def _stats(self,srcdir,targetdir):

		assert os.path.isdir(srcdir)
		if not os.path.isdir(targetdir):	os.mkdir(targetdir)

		gramfiles = glob.glob(srcdir+'/*')
		stats = [os.path.join(targetdir,f.split('/')[-1]) for f in gramfiles]
		
		for (grf,statf) in zip(gramfiles,stats):
			f = open(grf,'r',encoding='ISO-8859-1')
			words = f.read().split()
			count = Counter(words)
			print("no of unique ngrams in {} : {}".format(grf,len(count)))

			st = []
			for (key,val) in count.items():
				st.append('{:<10} {:>10} {:>10.7f}'.format(key,val,val/len(words)))
			st = '\n'.join(st)
			fl = open(statf,'w',encoding='ISO-8859-1')
			header = '{:<10} {:>10} {:>10}'.format(str(self.gram_sz)+'-gram','count','probability')

			fl.write(header)
			fl.write('\n')
			fl.write(st)
			fl.close()
			f.close()

		print("Stats prepared, see {} directory".format(targetdir))
		return self


	def _splitFiles(self,srcdir,targetdir,n):

		assert os.path.isdir(srcdir)
		if not os.path.isdir(targetdir):	os.mkdir(targetdir)
		
		gramfiles = glob.glob(srcdir+'/*')
		
		for (index,gf) in enumerate(gramfiles):
			file = open(gf,'r',encoding='ISO-8859-1')
			grams = file.read().split()

			l = len(grams)

			sz = l // n

			for i in range(n):
				print('{}/{}_{}'.format(targetdir,gf.split('/')[-1],i))
				fw =  open('{}/{}_{}'.format(targetdir,gf.split('/')[-1],i),'w',encoding='ISO-8859-1')
				fw.write(" ".join(grams[i*sz:(i+1)*sz]))
				fw.close()

			file.close()

		s = len(gramfiles)*n
		print("created {} files".format(s))
		return self