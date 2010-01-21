package dk.kvl.dina;

//The Viterbi algorithm: find the most probable state path producing
//the observed outputs x

public class Viterbi extends HMMAlgo {
	double[][] v;         // the matrix used to find the decoding
	// v[i][k] = v_k(i) = 
	// log(max(P(pi in state k has sym i | path pi)))
	Traceback2[][] B;     // the traceback matrix
	Traceback2 B0;        // the start of the traceback 

	public Viterbi(HMM hmm, String x) {
		super(hmm, x);
		final int L = x.length();
		v = new double[L+1][hmm.nstate];
		B = new Traceback2[L+1][hmm.nstate];
		v[0][0] = 0;                // = log(1)
		for (int k=1; k<hmm.nstate; k++)
			v[0][k] = Double.NEGATIVE_INFINITY; // = log(0)
		for (int i=1; i<=L; i++)
			v[i][0] = Double.NEGATIVE_INFINITY; // = log(0)
		for (int i=1; i<=L; i++)
			for (int ell=0; ell<hmm.nstate; ell++) {
				int kmax = 0;
				double maxprod = v[i-1][kmax] + hmm.loga[kmax][ell];
				for (int k=1; k<hmm.nstate; k++) {
					double prod = v[i-1][k] + hmm.loga[k][ell];
					if (prod > maxprod) {
						kmax = k;
						maxprod = prod;
					}
				}
				v[i][ell] = hmm.loge[ell][x.charAt(i-1)] + maxprod;
				B[i][ell] = new Traceback2(i-1, kmax);
			}
		int kmax = 0;
		double max = v[L][kmax];
		for (int k=1; k<hmm.nstate; k++) {
			if (v[L][k] > max) {
				kmax = k;
				max = v[L][k];
			}
		}
		B0 = new Traceback2(L, kmax);
	}

	public String getPath() {
		StringBuffer res = new StringBuffer();
		Traceback2 tb = B0;
		int i = tb.i, j = tb.j;
		while ((tb = B[tb.i][tb.j]) != null) {
			res.append(hmm.state[j]);
			i = tb.i; j = tb.j;
		}        
		return res.reverse().toString();
	}

	public void print(Output out) {
		for (int j=0; j<hmm.nstate; j++) {
			for (int i=0; i<v.length; i++)
				out.print(HMM.fmtlog(v[i][j]));
			out.println();
		}
	}
}


