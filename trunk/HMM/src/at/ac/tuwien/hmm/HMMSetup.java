package at.ac.tuwien.hmm;

import java.util.ArrayList;

import be.ac.ulg.montefiore.run.jahmm.ObservationDiscrete;
import be.ac.ulg.montefiore.run.jahmm.ObservationInteger;
import be.ac.ulg.montefiore.run.jahmm.Opdf;
import be.ac.ulg.montefiore.run.jahmm.OpdfDiscrete;

/**
 * Helper class for handling HMM generation.
 * @author Christof Schmidt
 *
 */
public class HMMSetup {
	public static final int INITIAL = 9999;

	String[] stateNames;
	double[][] transitionMatrix;
	String emissionSymbols ;
	double[][] emissionMatrix;
	double[] initialProbabilities;


	/** An HMM Setup. Uniform distribution for all matrices
    @param stateNames array of state names (except initial state)
    @param emissionSymbols string of emission names (one char per state)
    */
	public HMMSetup(String[] stateNames, String emissionSymbols) {
		this( stateNames, getRandomMatrix(stateNames.length,stateNames.length), 
				emissionSymbols,  getRandomMatrix(stateNames.length,emissionSymbols.length()), 
				getRandomArray(stateNames.length));
	}

	/** An HMM Setup. 
    @param stateNames array of state names (except initial state)
    @param transitionMatrix matrix of transition probabilities (except initial state)
    @param emissionSymbols string of emission names (one char per state)
    @param emissionMatrix matrix of emission probabilities
    */
	public HMMSetup(String[] stateNames, double[][] transitionMatrix, String emissionSymbols, double[][] emissionMatrix) {
		this( stateNames, transitionMatrix, emissionSymbols, emissionMatrix, getUniformArray(stateNames.length));
	}
	
	/** An HMM Setup.
    @param stateNames array of state names (except initial state)
    @param transitionMatrix matrix of transition probabilities (except initial state)
    @param emissionSymbols string of emission names (one char per state)
    @param emissionMatrix matrix of emission probabilities
    @param initialProbabilities array of the initial state properties
    
    */
	public HMMSetup(String[] stateNames, double[][] transitionMatrix, 
			String emissionSymbols, double[][] emissionMatrix,
			double[] initialProbabilities) {
		super();
		this.stateNames = stateNames;
		this.transitionMatrix = transitionMatrix;
		this.emissionSymbols = emissionSymbols;
		this.emissionMatrix = emissionMatrix;
		this.initialProbabilities = initialProbabilities;
	}
	
	/**
	 * Converts setup to DINA compatible HMM
	 * @return
	 */
	public dk.kvl.dina.HMM getDinaHMM() {
		return new dk.kvl.dina.HMM(stateNames,  transitionMatrix, emissionSymbols, emissionMatrix);
	}

	/**
	 * Converts setup to jahmm compatible HMM
	 * @return
	 */
	public be.ac.ulg.montefiore.run.jahmm.Hmm<ObservationDiscrete<Action>> getJaHMM() {
		java.util.List<Opdf<ObservationDiscrete<Action>>> opdfs = 
			new ArrayList<Opdf<ObservationDiscrete<Action>>>();
		for (double[] emission :getEmissionMatrix()) {
			opdfs.add(new OpdfDiscrete<Action>(Action.class, emission) );
		}
		be.ac.ulg.montefiore.run.jahmm.Hmm<ObservationDiscrete<Action>> hmm = 
			new be.ac.ulg.montefiore.run.jahmm.Hmm<ObservationDiscrete<Action>>(
				this.initialProbabilities, this.transitionMatrix, opdfs);
		return hmm;
	}

	public String[] getStateNames() {
		return stateNames;
	}

	public void setStateNames(String[] stateNames) {
		this.stateNames = stateNames;
	}

	public double[][] getTransitionMatrix() {
		return transitionMatrix;
	}

	public void setTransitionMatrix(double[][] transitionMatrix) {
		this.transitionMatrix = transitionMatrix;
	}

	public String getEmissionSymbols() {
		return emissionSymbols;
	}

	public void setEmissionSymbols(String emissionSymbols) {
		this.emissionSymbols = emissionSymbols;
	}

	public double[][] getEmissionMatrix() {
		return emissionMatrix;
	}

	public void setEmissionMatrix(double[][] emissionMatrix) {
		this.emissionMatrix = emissionMatrix;
	}

/**
 * Generate HMM based on the wikipedia example
 * http://en.wikipedia.org/wiki/Viterbi_algorithm#Example
 * @param i number of example
 * @return
 */
	public static HMMSetup getHMM(int i) {

		String[] state = {"Rainy ", "Sunny "};
		String esym = "WSC";

		if (i == 0) {
			double[][] amat = {{0.7, 0.3} , {0.4, 0.6}};
			double[][] emat = { {0.1,0.4,0.5}, {0.6,0.3,0.1} };
			HMMSetup hmm = new HMMSetup( state,  amat, esym, emat);		
			return hmm;
		} else if (i == 1) {
			double[][] amat = {{0.95, 0.05} , {0.1, 0.9}};
			double[][] emat = { {0.1,0.4,0.5}, {0.6,0.3,0.1} };
			HMMSetup hmm = new HMMSetup( state,  amat, esym, emat);		
			return hmm;
		} else if (i == 2){
			double[][] amat = {{0.5, 0.5} , {0.6, 0.4}};
			double[][] emat = { {0.1,0.4,0.5}, {0.6,0.3,0.1} };
			HMMSetup hmm = new HMMSetup( state,  amat, esym, emat);		
			return hmm;
		} else if (i == 3){
			double[][] amat = {{0.5, 0.5} , {0.5, 0.5}};
			double[][] emat = { {0.7,0.2,0.1}, {0.4,0.4,0.2} };
			HMMSetup hmm = new HMMSetup( state,  amat, esym, emat);		
			return hmm;
		} else if (i == 4){
			esym = "WC";
			double[][] amat = {{0.5, 0.5} , {0.5, 0.5}};
			double[][] emat = { {0.9,0.1}, {0.1,0.9} };
			HMMSetup hmm = new HMMSetup( state,  amat, esym, emat);		
			return hmm;
		} else if (i == INITIAL){
			HMMSetup hmm = new HMMSetup( state,esym);		
			return hmm;
		} else {
			return null;
		}
	}
	
	/**
	 * Create a uniformly filled matrix (sums up to 1)
	 */
	public static double[][] getUniformMatrix(int rows, int columns) {
		double[][] matrix = new double[rows][];
		for(int i=0; i<rows;i++) {
			matrix [i] = getUniformArray(columns); 
		}
		return matrix;
	}
	
	/**
	 * Create a uniformly filled array (sums up to 1)
	 */
	public static double[] getUniformArray(int size) {
		double[] d = new double[size];
		double value = 1.0/size;
		for (int i=0; i<size; i++) {
			d[i] = value;
		}
		return d;
	}
	
	/**
	 * Create a Random filled matrix (sums up to 1)
	 */
	public static double[][] getRandomMatrix(int rows, int columns) {
		double[][] matrix = new double[rows][];
		for(int i=0; i<rows;i++) {
			matrix [i] = getRandomArray(columns); 
		}
		return matrix;
	}
	/**
	 * Create a Random filled array (sums up to 1)
	 */	
	public static double[] getRandomArray(int n) { //taken from dina
		double[] ps = new double[n];
		double sum = 0;
		// Generate random numbers
		for (int i=0; i<n; i++) {
			ps[i] = Math.random();
			sum += ps[i];
		}
		// Scale to obtain a discrete probability distribution
		for (int i=0; i<n; i++) 
			ps[i] /= sum;
		return ps;
	}
	
}
