package at.ac.tuwien.hmm.training;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.Vector;

import javax.smartcardio.ATR;

import at.ac.tuwien.hmm.HMMClassifier;
import at.ac.tuwien.hmm.HMMHandler;
import at.ac.tuwien.hmm.HMMUtil;
import be.ac.ulg.montefiore.run.jahmm.Hmm;
import be.ac.ulg.montefiore.run.jahmm.Observation;
import be.ac.ulg.montefiore.run.jahmm.Opdf;
import be.ac.ulg.montefiore.run.jahmm.learn.BaumWelchLearner;

/**
 * Simple Training implementation. Only builds one random HMM and trains 
 * it with the training data. 
 * 
 * @author Christof Schmidt
 *
 * @param <O>
 */
@SuppressWarnings("hiding")
public class SimpleTrainer<O extends Observation> implements Trainer<O> {
	public boolean DISPLAY = false;
	
	
	private Random random = new Random();
	private Map<Integer, Hmm<O>> hmms;
	private int numClasses; 
	private int _stateCount; 
	private int attributeValuesCount;
	private int numAttributes;
	private HMMHandler<O> handler;
	
	/** for serialization */
	static final long serialVersionUID = -3481068294659183010L;

	
	public SimpleTrainer(int numClasses, int numAttributes, int stateCount,
			int attributeValuesCount, HMMHandler<O> handler) {
		this.numClasses = numClasses;
		this._stateCount = stateCount;
		this.attributeValuesCount = attributeValuesCount;
		this.handler = handler;
		this.numAttributes = numAttributes;		
	}

	public void setRandom(Random random) {
		this.random = random;
	}
	
	
	public void initHmms() {
		
		hmms = new TreeMap<Integer, Hmm<O>>();		
	    for (int classNo=0;classNo<numClasses; classNo++ ) {
			int noOfStates = _stateCount; 
			if (noOfStates == -1) {
				noOfStates = HMMUtil.getRandomStateCount(numAttributes,random);
			}
	    	List<Opdf<O>> opdfs = handler.createOdpf(noOfStates);
			double[][] transitionMatrix = getMatrix(noOfStates, noOfStates, random);
	    			    	
	    	Hmm<O> hmm = new Hmm<O>(HMMUtil.getRandomArray(noOfStates, random), transitionMatrix, opdfs);
	    	hmms.put(classNo, hmm);
	    }
    }

		
	public void trainHmms(Map<Integer, List<List<O>>> trainingInstancesMap, int accuracy) {
	    BaumWelchLearner learner = new BaumWelchLearner();

		learner.setNbIterations(accuracy); // "accuracy" - 
		for (int classNo:trainingInstancesMap.keySet()) {
			List<List<O>> trainingInstances  = trainingInstancesMap.get(classNo);
			Hmm<O> hmm = hmms.get(classNo);
	    	if (DISPLAY) System.out.println("UnTrained HMM No "+classNo+":\r\n"+hmm.toString());
	    	Hmm<O> trainedHmm = learner.learn(hmm, trainingInstances);
	    	hmms.put(classNo, trainedHmm);
	    	if (DISPLAY) System.out.println("Trained HMM No "+classNo+":\r\n"+trainedHmm.toString());

		}		
	}
	
	// Train HMM for a certain Class
	public Hmm<O> trainHmm(Map<Integer, List<List<O>>> trainingInstancesMap, int accuracy,int classNo) {
	    BaumWelchLearner learner = new BaumWelchLearner();
		learner.setNbIterations(accuracy);
		List<List<O>> trainingInstances  = trainingInstancesMap.get(classNo);
		Hmm<O> hmm = hmms.get(classNo);
    	if (DISPLAY) System.out.println("UnTrained HMM No "+classNo+":\r\n"+hmm.toString());
    	Hmm<O> trainedHmm = learner.learn(hmm, trainingInstances);
    	// Update trained HMM
    	hmms.put(classNo, trainedHmm);
    	if (DISPLAY) System.out.println("Trained HMM No "+classNo+":\r\n"+trainedHmm.toString());
    	
    	return trainedHmm;
		
	}

	public double[][] getMatrix(int rows, int columns, Random random) {
		double[][] matrix = new double[rows][];
		for(int i=0; i<rows;i++) {
			matrix [i] = getArray(columns, random); 
		}
		return matrix;
	}
	
	public double[] getArray(int size, Random random) {
		return HMMUtil.getRandomArray(size, random);
	}
	
	public Map<Integer, Hmm<O>> getHmms() {
		return hmms;
	}

	public double[][] getNominalEmissionMatrix(int stateCount) {
		return getMatrix(stateCount, attributeValuesCount, random);
	}

	public double[] getNumericMeanArray(double givenMean,int stateCount) {
		return HMMUtil.getHomogenArray(stateCount, givenMean);
	}

	public double[] getNumericVarianceArray(double givenVariance,int stateCount) {
		return HMMUtil.getHomogenArray(stateCount, givenVariance);
		
	}

	@Override
	public void setHmms(Map<Integer, Hmm<O>> hmms) {
		this.hmms = hmms;
		
	}

	// Create a completely new HMM by changing the number of states. If the new number of states are 
	@Override
	public int perturbate2(int classNo, Vector<Integer> tabuList) {		
		assert (hmms != null);		
		int newStateNb = HMMUtil.newStates(tabuList,numAttributes);
		
		List<Opdf<O>> opdfs = handler.createOdpf(newStateNb);
		double[][] transitionMatrix = getMatrix(newStateNb, newStateNb, random);
		Hmm<O> hmm = new Hmm<O>(HMMUtil.getRandomArray(newStateNb, random), transitionMatrix, opdfs);
		
		// update the new HMM
		hmms.put(classNo, hmm);
		return newStateNb;
	}

	// Randomly change the pi array
	@Override
	public void perturbate1(int classNo) {
		assert (hmms != null);
		Hmm<O> hmm = hmms.get(classNo);
		int nbStates = hmm.nbStates();
		
		//Generate new pi array
		double[] newPi = HMMUtil.getRandomArray(nbStates, random);
		
		//Set the new pi array
		for (int i = 0; i < newPi.length ; i++) {
			hmm.setPi(i, newPi[i]);
		}
		
		// update the new HMM
		hmms.put(classNo, hmm);
		
	}
	
	// Randomly change the transition matrix
	public void perturbate3(int classNo) {
		assert (hmms != null);
		Hmm<O> hmm = hmms.get(classNo);
		int nbStates = hmm.nbStates();
		double[][] newTransitionMatrix = getMatrix(nbStates, nbStates, random);
		for (int i = 0; i < nbStates; i++) {
			for (int j = 0; j < nbStates; j++) {
				hmm.setAij(i, j, newTransitionMatrix[i][j]);
			}
		}
		
		hmms.put(classNo, hmm);
	}

	@Override
	public Hmm<O> getHmm(int classNo) {
		return this.hmms.get(classNo);
	}


	
}
