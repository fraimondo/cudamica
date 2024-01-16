#!/usr/bin/env python

import os, stat, sys, re

IGNORE_LIST = ['.svn', 'bin', 'deleter.py']
IGNORE_EXT = ['.o', '.bin']
KEYWORDS = ['DATATEST', 'STEPDATATEST', 'BLOWUPTEST', 'STEPDUMP', 'BIASTEST']

def walktree(top="", depthfirst=False):
	try:
		path = top if top else '.'
		#~ print path
		tst = os.lstat(path)
		#~ print tst
	except os.error:
		tst = None
	if not depthfirst:
		yield tst,top

	if stat.S_ISDIR(tst.st_mode):
		names = os.listdir(top if top else '.')
		for name in names:
			try:
				st = os.lstat(os.path.join(top, name) if top else name)
			except os.error:
				continue
			newtop = os.path.join(top, name) if top else name
			if any(name.endswith(ext) for ext in IGNORE_EXT):
				continue
			if name not in IGNORE_LIST:
				for x in walktree (newtop, depthfirst):
					yield x
	if depthfirst:
		yield tst,top

IF_START  = r'.*#if(def)?.*'
IFN_START  = r'.*#ifndef.*'
IF_END	  = r'.*#endif.*'
DEL_START = r'.*#ifdef\s*(' + '|'.join(KEYWORDS) + ').*'
DEL_START_TOGGLE = r'.*#ifndef\s*(' + '|'.join(KEYWORDS) + ').*'
EL_END   = r'.*#else.*'

def mustdelete(levels):
	return any(levels)

def deleterto(src, dst):
	#~ print DEL_START
	#~ return
	if not os.path.exists(dst):
		os.makedirs(dst, mode=0755)
	#~ print srcbase
	for st,fn in walktree(src):
		#~ print fn
		npath = fn[len(src):]
	
		nfn = os.path.join(dst, npath)
		#~ print nfn
		#~ print fn
		if stat.S_ISDIR(st.st_mode):
			if not os.path.exists(nfn):
				os.mkdir(nfn, 0755)
		else:
			levels = [False]
			levelmatters = [False]
			depth = 0
			nf = open(nfn, 'w')
			nl = 0
			for l in open(fn, 'r'):
				#~ print l
				nl += 1
				if re.match(IF_START, l, re.IGNORECASE):
					depth +=1
					levels.append(False)
					levelmatters.append(False)
					#~ print "DEBUG::Nested IN " +  str(depth)
					if re.match(DEL_START, l, re.IGNORECASE):
						#~ print "DEBUG::START deleting"
						levelmatters[depth] = True
						levels[depth] = True
				if re.match(IFN_START, l, re.IGNORECASE):
					depth +=1
					levels.append(False)
					levelmatters.append(False)
					#~ print "DEBUG::Nested IN " +  str(depth)
					if re.match(DEL_START_TOGGLE, l, re.IGNORECASE):
						#~ print "DEBUG::START deleting"
						levelmatters[depth] = True
						levels[depth] = False						
				if (not (mustdelete(levels) or 
					(re.match(IF_END, l, re.IGNORECASE) and levelmatters[depth]) or 
					(re.match(IFN_START, l, re.IGNORECASE) and levelmatters[depth]) or 
					(re.match(EL_END, l, re.IGNORECASE) and levelmatters[depth]))):
					#~ print "DEBUG::Deleting= " + l
				#~ else:
					nf.write(l)
					#~ print "DEBUG::Bypassing= " + l
				if re.match(EL_END, l, re.IGNORECASE) and levelmatters[depth]:
					levels[depth] = not levels[depth]
					#~ print "DEBUG::Toggle at " + l
				if re.match(IF_END, l, re.IGNORECASE):
					#~ print "DEBUG::Nested OUT " +  str(depth)
					levels.pop(depth)
					levelmatters.pop(depth)
					depth -= 1
						
				
				

			del nf
			#~ 
			

def main(argv):
	#~ deleterto('tmp/cudaica-426/src/', './tmp/cudaica-clean/')
	deleterto(argv[1], argv[2])

if __name__ == "__main__":
	main(sys.argv)
