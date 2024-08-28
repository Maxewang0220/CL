#!/usr/bin/python3



import sys

if len(sys.argv) != 3:
	sys.exit("\nUsage: " + sys.argv[0] + " <gold file> <tagger file>\n")

# open the tagger output file and the gold standard file
file_gold = open(sys.argv[1])
file_system = open(sys.argv[2])

precision_recall = {}

# read from files
n = 0
for line_gold, line_system in zip(file_gold, file_system):

	line_gold = line_gold.rstrip()
	line_system = line_system.rstrip()
	n = n+1

	if len(line_gold) != 0:

		word_gold, tag_gold = line_gold.split("\t")
		word_system, tag_system = line_system.split("\t")

		# word forms should match in gold and system file
		if word_system != word_gold:
			sys.exit("\nError in line " + str(n) + ": word mismatch!\n")

		if precision_recall.get(tag_system) == None: precision_recall[tag_system] = [0, 0, 0]
		if precision_recall.get(tag_gold) == None: precision_recall[tag_gold] = [0, 0, 0]

		precision_recall[tag_system][1] += 1 # tag was assigned by system
		precision_recall[tag_gold][2] += 1 # tag was found in gold standard data

		# observe and count correct tags
		if tag_system == tag_gold:
			precision_recall[tag_gold][0] += 1 # tag assignment was correct

# counts for overall accuracy
correct = 0
overall = 0

print("\nComparing gold file \"" + sys.argv[1] + "\" and system file \"" + sys.argv[2] + "\"")
print("\nPrecision, recall, and F1 score:\n")

for tag, counts in precision_recall.items():

	# calculate precision, recall and F1 score, print them
	correct += counts[0]
	overall += counts[1]

	# precision, recall, and f1 for a subset of tags
	if not 0 in counts:
		precision = counts[0] / counts[1]
		recall = counts[0] / counts[2]
		f1_score = (2 * precision * recall) / (precision + recall)
		print("%5s %.4f %.4f %.4f" % (tag, precision, recall, f1_score))

print("\nAccuracy: %.4f\n" % (correct / overall))
