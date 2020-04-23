import	somfft_beta, sys, math, argparse

def	read_data(file):
	f = open(file, "r")
	x = f.readlines()
	f.close()
	D = []
	Label = []
	ln = len(x)
	for i in range(ln):
		tmp = []
		xx = x[i].split('\t')
		na = len(xx)-1
		if xx[0][0] != "#":
			for j in range(na):
				tmp.append(float(xx[j]))
			D.append(tmp)
			Label.append(xx[len(xx) - 1][0:len(xx[len(xx) - 1]) - 1])
	return [D, Label]

def	read_attr(file):
	f = open(file, "r")
	x = f.readlines()
	f.close()
	A = []
	for i in range(len(x)):
		A.append(int(x[i]))
	return A


"""
"""
print "Reading data"

parser = argparse.ArgumentParser()
parser.add_argument('-i', action = "store", dest = "i", help = "The input file. Last column is the label")
parser.add_argument('-e', action = "store", dest = "e", help = "Number of epochs")
parser.add_argument('-m', action = "store", dest = "M", help = "Number of rows in the lattice")
parser.add_argument('-n', action = "store", dest = "N", help = "Number of columns in the lattice")
parser.add_argument('-wt', action = "store", dest = "wt", help = "The output file containing the trainned weights")
parser.add_argument('-mp', action = "store", dest = "mp", help = "The output file containing the mapping")
parser.add_argument('-bmu', action = "store", dest = "bmu", help = "The output file containing the BMUs. It is a NxM matrix. N data, M epochs")
parser.add_argument('-c', action = "store", dest = "c", help = "The type of training (som,fft)")
parser.add_argument('-stats', action = "store", dest = "stats", help = "The output file containing some statistics such as te, eq")
parser.add_argument('-vec', action = "store", dest = "vecStats", help = "The output file containing statistics about the vectors (trajectories in the input space etc)")
parser.add_argument('-bs', action = "store", dest = "bmuStats", help = "The output file containing statistics about the BMUs (trajectories in the weight space, number of times selected as BMU, etc)")
parser.add_argument('-HF', action = "store", dest = "HF", help = "The neighborhood of the FFT unit")
parser.add_argument('-bD', action = "store", dest = "bD", help = "Distance threshold in terms of percentage")


#parser.add_argument('-', action = "store", dest = "", help = "")

args = parser.parse_args()



[D, Label] = read_data(args.i)

M = int(args.M)
N = int(args.N)
dim = len(D[0])
epochs = int(args.e)
print "Creating network"
type_training = args.c
ss = somfft.SOM(M, N, M - 1, 0, 0.1, 0.00001, dim, epochs, type_training, args.wt)


print "ss = ", ss.epochs
#ss = som.SOM(M, N, M + N, 0, 0.1, 0.00001, dim, epochs)
ss.init_weight()
ss.obtain_H()
ss.write_weights("tmp_0x98abb_1.csv")
if args.HF == True:
	ss.HF = int(args.HF)
else:
	ss.HF = 1

print "Training"
ss.bD = float(args.bD)

ss.training(D)

[Map, Wt_BMU] = ss.mapping(D)
#print M
eq = ss.err_quant(D)
print "eq = ", eq
te = ss.top_err(D)
print "te = ", te

ss.write_weights(args.wt)

ss.write_map(Map, Label, args.mp)

ss.write_bmus(D, args.bmu)

ss.write_quality_stats(te, eq, args.stats)
ss.write_stats_vectors(D, args.vecStats)
ss.write_stats_BMUs(args.bmuStats)
