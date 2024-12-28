import os

# number of genomes
max_fr = 12811

# list of mutation freqs
with open('uniq.tsv') as info:
	rare = info.readlines()

# dict with mutation frequencies
d = dict()
for r in rare:
	tmp = r.split()
	d['#'.join(tmp[1:])] = int(tmp[0])

# minimum frequency
thr = 5

for f in os.listdir('translated_no_triplets_and_clips'):
	#print(f)
	with open('translated_no_triplets_and_clips/' + f) as info:
		lines = info.readlines()
	# output
	with open('feature_lists/' + f, 'w') as out:
		for l in lines:
			ft = '#'.join(l.split())
			# if the feature is too rare or too frequent: skip
			if d[ft] < thr or d[ft] > max_fr - thr:
				continue
			print(ft, end=' ', file=out)
	#break