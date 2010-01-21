package at.ac.tuwien.hmm;

import java.util.ArrayList;
import java.util.Random;

import be.ac.ulg.montefiore.run.jahmm.ObservationDiscrete;
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

	private static Random random = new Random();

	/** An HMM Setup. Uniform distribution for all matrices
    @param stateNames array of state names (except initial state)
    @param emissionSymbols string of emission names (one char per state)
    */
	public HMMSetup(String[] stateNames, String emissionSymbols) {
		this( stateNames, HMMUtil.getRandomMatrix(stateNames.length,stateNames.length,random), 
				emissionSymbols,   
				HMMUtil.getRandomMatrix(stateNames.length, emissionSymbols.length(),random), 
				HMMUtil.getRandomArray(stateNames.length,random));
	}

	/** An HMM Setup. 
    @param stateNames array of state names (except initial state)
    @param transitionMatrix matrix of transition probabilities (except initial state)
    @param emissionSymbols string of emission names (one char per state)
    @param emissionMatrix matrix of emission probabilities
    */
	public HMMSetup(String[] stateNames, double[][] transitionMatrix, String emissionSymbols, double[][] emissionMatrix) {
		this( stateNames, transitionMatrix, emissionSymbols, 
				emissionMatrix, HMMUtil.getUniformArray(stateNames.length));
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

	
}
