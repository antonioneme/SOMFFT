import	random, sys, math
import	numpy as np
#import	scipy as sp
from scipy import spatial

class	SOM:

	def	__init__(self, N, M, H_i, H_f, a_i, a_f, dim, epochs, tt):
		self.N = N
		self.M = M
		self.H_i = H_i
		self.H_f = H_f
		self.a_i = a_i
		self.a_f = a_f
		self.dim = dim
		self.epochs = epochs
		self.tt = tt

	def	init_weight(self):
		# rows
		random.seed()
		self.weight = [None] * self.N
		self.weight_prev = [None] * self.N
		# columns
		for i in range(self.N):
			self.weight[i] = [None] * self.M
			self.weight_prev[i] = [None] * self.M

		for i in range(self.N):
			for j in range(self.M):
				self.weight[i][j] = [float] * self.dim
				# weights
				for k in range(self.dim):
					self.weight[i][j][k] = random.random()

		# Compute the distance matrix of all units in the lattice
		self.Lattice = []
		for i in range(self.N):
			for j in range(self.M):
				self.Lattice.append([i,j])
		#print "M = ", self.Lattice
		self.DMx = spatial.distance.pdist(self.Lattice, metric='euclidean')
		self.DM = spatial.distance.squareform(self.DMx)
		#print "DM = ", self.DM[0], self.DM[3]

		# BMU_T[(v,t)] is the BMU for vector v at epoch t
		self.BMU_T = {}


	def	obtain_H(self):
		self.cte_H = math.log( ( float(self.H_f) + 0.0001)/ float(self.H_i) ) / math.log(2.718281);
		self.cte_a = math.log( ( float(self.a_f) + 0.0001)/ float(self.a_i) ) / math.log(2.718281);

	# The distance between the weight vector and the input data
	def	dist(self, weight, dato):
		d = 0.0
		for k in range(self.dim):
			d = d + abs(weight[k] - dato[k])
		return d


	# se obtiene la BMU para el dato i
	def	BMU(self, dato):
		dmin = 10000000000000.0
		for i in range(self.N):
			for j in range(self.M):
				#print "dato = ", dato
				#print "wt = ", self.weight[i][j]
				d = self.dist(self.weight[i][j], dato)
				if d <= dmin:
					dmin = d
					cual = [i, j]

		return cual

	# Obtain the 2nd BMU for vector i
	def	BMUX(self, dato, bmu):
		dmin = 10000000000000.0

		for i in range(self.N):
			for j in range(self.M):
				#if i != bmu[0] and j != bmu[1]:
				#if [i, j] != bmu:
				if i != bmu[0] or j != bmu[1]:
					d = self.dist(self.weight[i][j], dato)
					if d <= dmin:
						dmin = d
						cual = [i, j]

		return cual



	def	adaptation(self, dato, bmu, H, a):
		# An unit can be affected even if it is not a BMU:
		"""
		for i in range(bmu[0] - H/2, bmu[0] + H/2 + 1):
			for j in range(bmu[1] - H/2, bmu[1] + H/2 + 1):
				if i >= 0 and i < self.N and j >= 0 and j < self.M:
					self.weight_prev[i][j] = list(self.weight[i][j])
		"""
		self.weight_prev[bmu[0]][bmu[1]] = list(self.weight[bmu[0]][bmu[1]])
	
		for i in range(bmu[0] - H/2, bmu[0] + H/2 + 1):
			for j in range(bmu[1] - H/2, bmu[1] + H/2 + 1):
				if i >= 0 and i < self.N and j >= 0 and j < self.M:
					for k in range(self.dim):
						#self.weight[i][j][k] = self.weight[i][j][k] + a * H * (dato[k] - self.weight[i][j][k])
						self.weight[i][j][k] = self.weight[i][j][k] + a * (dato[k] - self.weight[i][j][k])



	def	BMUFFT(self, dato, HF):
		dmin = 10000000000000.0

		# MDB is the Most Distant BMU
		# MDB is the distance between each BMU in the list Selected
		# and the BMUs not Selected so far, that is, the Available BMUs
		#print "A = ", len(self.Selected), len(self.Available), self.Selected

		# MDB[row] is len(self.Selected)
		MDB = spatial.distance_matrix(self.Selected, self.Available)
		#print "M = ", len(MDB), len(MDB[0])
		#print "M = ", len(MDB), len(MDB[0]), MDB[0], MDB[1]
		# MDB_D[i] is the total distance between the available BMUs and the
		# already selected BMUs
		MDB_D = {}
		# Here you are, AN
		for i, b in enumerate(self.Available):
			#print "b = ", b, i
			MDB_D[(b[0],b[1])] = 0.0
			for j in range(len(MDB)):
				#print "j = ", j
				MDB_D[(b[0],b[1])] = MDB_D[(b[0],b[1])] + MDB[j][i]

		dmax = 0.0
		for a in self.Available:
			if MDB_D[(a[0],a[1])] >= dmax:
				dmax = MDB_D[(a[0],a[1])]
				candBMU = a

		# Select the BMU located at the farthest distance from all
		# selected BMUs. Use average distance?
		# Select the BMU from a small vicinity from k
		dmin = 1000000000.0
		for i in range(candBMU[0] - HF, candBMU[0] + HF + 1):
			for j in range(candBMU[1] - HF, candBMU[1] + HF + 1):
				if i >= 0 and i < self.N and j >= 0 and j < self.M:
					if [i,j] not in self.Selected:
						d = self.dist(self.weight[i][j], dato)
						if d <= dmin:
							dmin = d
							which = [i, j]
		#print "w = ", which
		#print "Sel = ", self.Selected
		#print "Av = ", self.Available
		self.Selected.append(which)
		self.Available.remove(which)

		return which


	def	BMUFFTX(self, HF):
		dmin = 10000000000000.0

		if self.Available == []:
			return []
		# MDB is the Most Distant BMU
		# MDB is the distance between each BMU in the list Selected
		# and the BMUs not Selected so far, that is, the Available BMUs
		#print "A = ", len(self.Selected), len(self.Available), self.Selected
		#print "A = ", len(self.Selected), len(self.Available)

		# MDB[row] is len(self.Selected)
		MDB = spatial.distance_matrix(self.Selected, self.Available)
		#print "M = ", len(MDB), len(MDB[0])
		#print "M = ", len(MDB), len(MDB[0]), MDB[0], MDB[1]
		# MDB_D[i] is the total distance between the available BMUs and the
		# already selected BMUs
		MDB_D = {}
		for i, b in enumerate(self.Available):
			#print "b = ", b, i
			MDB_D[(b[0],b[1])] = 0.0
			for j in range(len(MDB)):
				#print "j = ", j
				MDB_D[(b[0],b[1])] = MDB_D[(b[0],b[1])] + MDB[j][i]

		dmax = 0.0
		for a in self.Available:
			if MDB_D[(a[0],a[1])] >= dmax:
				dmax = MDB_D[(a[0],a[1])]
				candBMU = a

		# Select the BMU located at the farthest distance from all
		# selected BMUs. Use average distance?
		# Select the BMU from a small vicinity from k
		dmin = 1000000000.0
		candRegion = []
		for i in range(candBMU[0] - HF, candBMU[0] + HF + 1):
			for j in range(candBMU[1] - HF, candBMU[1] + HF + 1):
				if i >= 0 and i < self.N and j >= 0 and j < self.M:
					if [i,j] not in self.Selected:
						candRegion.append([i,j])

		return candRegion


	def	BMUrand(self, HF):
		dmin = 10000000000000.0

		if self.Available == []:
			return []
		# Return a random unit from the available ones

		k = int( random.random() * len(self.Available) )
		candBMU = self.Available[k]
		dmin = 1000000000.0
		candRegion = []
		for i in range(candBMU[0] - HF, candBMU[0] + HF + 1):
			for j in range(candBMU[1] - HF, candBMU[1] + HF + 1):
				if i >= 0 and i < self.N and j >= 0 and j < self.M:
					if [i,j] not in self.Selected:
						candRegion.append([i,j])

		return candRegion
		

	# The description of vectors that are available
	def	vector_desc(self, datos, vecAvailable):
		VAW = []
		for v in vecAvailable:
			VAW.append(datos[v])
		return VAW


	# The weight vector of the available units
	def	unit_desc(self, candBMUs):
		UW = []
		for b in candBMUs:
			UW.append(self.weight[b[0]][b[1]])
		return UW

	def	training(self, datos):
		H = self.H_i
		a = self.a_i
		#HF = 1
		#HF = 3
		HF = self.HF
		ln = len(datos)

		# Distance traveled for each vector: The total distance between BMUs
		# for all the training process 
		self.Dist_V_T = {}

		for v in range(ln):
			self.Dist_V_T[v] = 0.0

		# The number of times each unit was selected as BMU
		self.numBMU = {}
		# The distance followed by the trajectory, in the weight space, for each BMU
		self.BMU_Dist = {}
		for b in self.Lattice:
			self.numBMU[(b[0],b[1])] = 0
			self.BMU_Dist[(b[0],b[1])] = 0.0

		for t in range(self.epochs):
			self.Selected = []
			self.Available = list(self.Lattice)
			if self.tt == "fft" or self.tt == "som":
				for i in range(ln):
					#print "ii = ", datos[i]
					if self.tt == "fft":
						if self.Available == []:
						# If all units have been selected, restart
							self.Available = list(self.Lattice)
							self.Selected = []
						if self.Selected == []:
							bmu = self.BMU(datos[i])
							self.Selected.append(bmu)
							self.Available.remove(bmu)
						else:
							bmu = self.BMUFFT(datos[i], HF)
							#print "i = ", i, bmu, self.Selected
							#cc = sys.stdin.read(1)
					else:
						if self.tt == "som":
							bmu = self.BMU(datos[i])

					self.adaptation(datos[i], bmu, int(H), a)
					self.BMU_T[(i,t)] = bmu
					if t > 0:
						vv1 = abs(bmu[0] - self.BMU_T[(i,t-1)][0])
						vv2 = abs(bmu[1] - self.BMU_T[(i,t-1)][1])
						self.Dist_V_T[i] = self.Dist_V_T[i] + (vv1 + vv2)
					self.numBMU[(bmu[0],bmu[1])] = self.numBMU[(bmu[0],bmu[1])] + 1
					#print "bmu = ", bmu
					#print "bb = ", self.weight[bmu[0]][bmu[1]]
					#print "bb = ", self.weight_prev[bmu[0]][bmu[1]]
					#cc = sys.stdin.read(1)
					self.BMU_Dist[(bmu[0],bmu[1])] = self.BMU_Dist[(bmu[0],bmu[1])] + self.dist(self.weight[bmu[0]][bmu[1]], self.weight_prev[bmu[0]][bmu[1]])
			else:
				if self.tt == "fftx":
					# The real FFT algorithm
					# The already mapped vectors
					vecSel = []
					
					vecAvailable = list(range(ln))
					# The first vector to be mapped
					k = int(random.random()*ln)
					bmu = self.BMU(datos[k])
					self.adaptation(datos[k], bmu, int(H), a)
					# BMUs
					self.Selected.append(bmu)
					self.Available.remove(bmu)
					# Vectors
					vecSel.append(k)
					vecAvailable.remove(k)
					self.BMU_T[(k,t)] = bmu

					#print "SB = ", self.Selected
					#print "AB = ", self.Available
					#print "SV = ", vecSel
					#print "AV = ", vecAvailable

					while len(vecSel) < ln:
						#print "vS = ", vecSel, len(vecSel)
						# candBMUs is the list containing the candidate
						# BMUs, based on a FFT condition
						candBMUs = self.BMUFFTX(HF)
						#print "cB = ", candBMUs
						if candBMUs != []:
							# The description of vectors that are available
							VAW = self.vector_desc(datos, vecAvailable)
							# The weight vector of the available units
							UW = self.unit_desc(candBMUs)
							# Select the vector that is closest to
							# any of the candidate BMUs
							# Distance matrix from vectors to units
							DMVB = spatial.distance_matrix(VAW, UW)

							# From the available vectors, select the closest
							# to the candidate BMUs
							minD = 1000000000.0
							wV = -1
							wBMU = -1
							for i,v in enumerate(vecAvailable):
								for j in range(len(DMVB[i])):
									if DMVB[i][j] < minD:
										minD = DMVB[i][j]
										# The "winner" vector
										wV = v
										# The BMU
										wBMU = j

							bmu = candBMUs[wBMU]

						else:
							# If all units have been selected, restart
							self.Available = list(self.Lattice)
							self.Selected = []
							# Select a vector from the available ones
							wV = vecAvailable[int(random.random()*len(vecAvailable))]
							bmu = self.BMU(datos[wV])

						vecSel.append(wV)
						vecAvailable.remove(wV)
						self.Selected.append(bmu)
						self.Available.remove(bmu)

						self.adaptation(datos[wV], bmu, int(H), a)
						self.BMU_T[(wV,t)] = bmu
						#print "t = ", t, wV
						if t > 0:
							vv1 = abs(bmu[0] - self.BMU_T[(wV,t-1)][0])
							vv2 = abs(bmu[1] - self.BMU_T[(wV,t-1)][1])
							self.Dist_V_T[wV] = self.Dist_V_T[wV] + (vv1 + vv2)
						self.numBMU[(bmu[0],bmu[1])] = self.numBMU[(bmu[0],bmu[1])] + 1
						self.BMU_Dist[(bmu[0],bmu[1])] = self.BMU_Dist[(bmu[0],bmu[1])] + self.dist(self.weight[bmu[0]][bmu[1]], self.weight_prev[bmu[0]][bmu[1]])

					#print "epoch = ", t, len(vecSel)
				else:
					if self.tt == "rBMU":
						# The real FFT algorithm
						# The already mapped vectors
						vecSel = []
						
						vecAvailable = list(range(ln))
						# The first vector to be mapped
						k = int(random.random()*ln)
						bmu = self.BMU(datos[k])
						self.adaptation(datos[k], bmu, int(H), a)
						# BMUs
						self.Selected.append(bmu)
						self.Available.remove(bmu)
						# Vectors
						vecSel.append(k)
						vecAvailable.remove(k)
						self.BMU_T[(k,t)] = bmu
	
						#print "SB = ", self.Selected
						#print "AB = ", self.Available
						#print "SV = ", vecSel
						#print "AV = ", vecAvailable
	
						while len(vecSel) < ln:
							#print "vS = ", vecSel, len(vecSel)
							# candBMUs is the list containing the candidate
							# BMUs, based on a FFT condition
							candBMUs = self.BMUrand(HF)
							#print "cB = ", candBMUs
							if candBMUs != []:
								# The description of vectors that are available
								VAW = self.vector_desc(datos, vecAvailable)
								# The weight vector of the available units
								UW = self.unit_desc(candBMUs)
								# Select the vector that is closest to
								# any of the candidate BMUs
								# Distance matrix from vectors to units
								DMVB = spatial.distance_matrix(VAW, UW)
	
								# From the available vectors, select the closest
								# to the candidate BMUs
								minD = 1000000000.0
								wV = -1
								wBMU = -1
								for i,v in enumerate(vecAvailable):
									for j in range(len(DMVB[i])):
										if DMVB[i][j] < minD:
											minD = DMVB[i][j]
											# The "winner" vector
											wV = v
											# The BMU
											wBMU = j
	
								bmu = candBMUs[wBMU]
	
							else:
								# If all units have been selected, restart
								self.Available = list(self.Lattice)
								self.Selected = []
								# Select a vector from the available ones
								wV = vecAvailable[int(random.random()*len(vecAvailable))]
								bmu = self.BMU(datos[wV])
	
							vecSel.append(wV)
							vecAvailable.remove(wV)
							self.Selected.append(bmu)
							self.Available.remove(bmu)
	
							self.adaptation(datos[wV], bmu, int(H), a)
							self.BMU_T[(wV,t)] = bmu
							#print "t = ", t, wV
							if t > 0:
								vv1 = abs(bmu[0] - self.BMU_T[(wV,t-1)][0])
								vv2 = abs(bmu[1] - self.BMU_T[(wV,t-1)][1])
								self.Dist_V_T[wV] = self.Dist_V_T[wV] + (vv1 + vv2)
							self.numBMU[(bmu[0],bmu[1])] = self.numBMU[(bmu[0],bmu[1])] + 1
							self.BMU_Dist[(bmu[0],bmu[1])] = self.BMU_Dist[(bmu[0],bmu[1])] + self.dist(self.weight[bmu[0]][bmu[1]], self.weight_prev[bmu[0]][bmu[1]])
					

			H = H * math.exp(float(t) * self.cte_H / float(self.epochs))
			a = a * math.exp(float(t) * self.cte_a / float(self.epochs))
			if t % 10 == 0:
				print "epoch = ", t
				print "H = ", H, a
			if H < 0.00000001 or a < 0.00000001:
				self.epoch_F = t
				break

	def	write_weights(self, arch):
		f = open(arch, "w")
		for i in range(self.N):
			for j in range(self.M):
				for k in range(self.dim-1):
					f.write(str(self.weight[i][j][k]))
					f.write("\t")
					#f.write(" ")
				f.write(str(self.weight[i][j][self.dim-1]))
				f.write("\n")
		f.close


	def	write_bmus(self, D, arch):
		ln = len(D)
		f = open(arch, "w")
		f.write("#ID\tepoch\tBMU\tBMUList\n")
		for i in range(ln):
			for t in range(self.epoch_F):
				v = self.BMU_T[(i,t)][0] * self.N + self.BMU_T[(i,t)][1]
				f.write( str(i) + "\t" + str(t) + "\t" + str(v) + "\t" + str(self.BMU_T[(i,t)]) + "\n")
		f.close()

	def	write_quality_stats(self, te, eq, arch):
		f = open(arch, "w")
		f.write(str(te) + "\t" + str(eq) + "\n")
		f.close()

	def	write_stats_vectors(self, D, arch):
		ln = len(D)
		f = open(arch, "w")
		f.write("#ID\tDist\tFixEpoch\n")
		for i in range(ln):
			vv = -1
			# The epoch at which the BMU fixed for vector i
			for j in range(self.epoch_F-1, 0, -1):
				if self.BMU_T[(i,j)] != self.BMU_T[(i,j+1)]:
					vv = j
					break
			f.write( str(i) + "\t" + str(self.Dist_V_T[i]) + "\t" + str(vv) + "\n" )
		f.close()

	def	write_stats_BMUs(self, arch):
		f = open(arch, "w")
		f.write("#ID\tnum\tDist\tIDList\n")
		for b in self.Lattice:
			vv = b[0] * self.N + b[1]
			f.write( str(vv) + "\t" + str(self.numBMU[(b[0],b[1])]) + "\t" + str(self.BMU_Dist[(b[0],b[1])]) + "\t" + str(b) + "\n"  )
		f.close()

	def	mapping(self, datos):
		Mapa = []
		Ps = []
		for i in range(len(datos)):
			bmu = self.BMU(datos[i])
			Mapa.append(bmu)
			Ps.append(self.weight[bmu[0]][bmu[1]])
		return [Mapa, Ps]

	def	write_map(self, Mapa, Label, arch):
		f = open(arch, "w")
		for i in range(len(Mapa)):
			f.write(str(Mapa[i][0]))
			f.write("\t")
			#f.write(" ")
			f.write(str(Mapa[i][1]))
			#f.write(" ")
			f.write("\t")
			f.write(Label[i])
			f.write("\n")
		f.close

	def	write_weight_mapa(self, Ps, arch):
		f = open(arch, "w")
		for i in range(len(Ps)):
			for k in range(self.dim):
				f.write(str(Ps[i][k]))
				f.write("\t")
				#f.write(" ")
			f.write("\n")
		f.close
	
	def	err_quant(self, dats):
		EQ = 0.0
		ln = len(dats)
		for i in range(ln):
			bmu = self.BMU(dats[i])
			e = self.dist(dats[i], self.weight[bmu[0]][bmu[1]])
			EQ = EQ + e
		EQ = EQ / float(ln)

		return EQ

	def	dist2(self, a, b):
		d = 0.0
		for k in range(2):
			d = d + abs(float(a[k]) - float(b[k]))
		return d
		
		
	def	top_err(self, dats):
		TE = 0.0
		ln = len(dats)
		for i in range(ln):
			bmu = self.BMU(dats[i])
			bmux = self.BMUX(dats[i], bmu)
			e = self.dist2(bmu, bmux)
			#print "bmu = ", bmu, " bmux = ", bmux, " e = ", e
			#if e > 1.0:
			#	TE = TE + 1.0
			TE = TE + e
		#TE = TE / float(ln)
		TE = TE / ( (self.N + self.M) * float(ln) )

		return TE

	def	topographic_error(self, dats):
		TE = 0.0
		ln = len(dats)
		for i in range(ln):
			bmu = self.BMU(dats[i])
			bmux = self.BMUX(dats[i], bmu)
			e = self.dist2(bmu, bmux)
			#print "bmu x = ", bmu, " bmux = ", bmux, " e = ", e
			if e > 1.0:
				TE = TE + 1.0
		#TE = TE / float(ln)
		TE = TE / ( (self.N + self.M) * float(ln) )

		return TE



# FFT:
# Obtained from:
# https://gist.github.com/nkt1546789/8e6c46aa4c3b55f13d32
import numpy as np

def fft(X,D,k):
    """
    X: input vectors (n_samples by dimensionality)
    D: distance matrix (n_samples by n_samples)
    k: number of centroids
    out: indices of centroids
    """
    n=X.shape[0]
    visited=[]
    i=np.int32(np.random.uniform(n))
    visited.append(i)
    while len(visited)<k:
        dist=np.mean([D[i] for i in visited],0)
        for i in np.argsort(dist)[::-1]:
            if i not in visited:
                visited.append(i)
                break
    return np.array(visited)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    n=300
    e=0.1
    mu1=np.array([-2.,0.])
    mu2=np.array([2.,0.])
    mu3=np.array([0.,2.])
    mu=np.array([mu1,mu2,mu3])
    x1=np.random.multivariate_normal(mu1,e*np.identity(2),n/3)
    x2=np.random.multivariate_normal(mu2,e*np.identity(2),n/3)
    x3=np.random.multivariate_normal(mu3,e*np.identity(2),n/3)
    X=np.r_[x1,x2,x3]
    y=np.concatenate([np.repeat(0,int(n/3)),np.repeat(1,int(n/3)),np.repeat(2,int(n/3))])

    X2=np.c_[np.sum(X**2,1)]
    D=X2+X2.T-2*X.dot(X.T)
    centroid_idx=fft(X,D,3)
    centroids=X[centroid_idx]

    colors=plt.cm.Paired(np.linspace(0, 1, len(np.unique(y))))
    plt.scatter(X[:,0],X[:,1],color=colors[y])
    plt.scatter(centroids[:,0],centroids[:,1],color="black",s=50)
    plt.show()
