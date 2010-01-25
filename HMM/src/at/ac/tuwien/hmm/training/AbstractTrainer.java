package at.ac.tuwien.hmm.training;

import java.util.Map;
import java.util.Random;

import at.ac.tuwien.hmm.HMMHandler;
import at.ac.tuwien.hmm.HMMUtil;
import be.ac.ulg.montefiore.run.jahmm.Hmm;
import be.ac.ulg.montefiore.run.jahmm.Observation;

/**
 * Basic methods for Trainer implementations.
 * 
 * @author Christof Schmidt
 *
 * @param <O>
 */
public abstract class AbstractTrainer<O extends Observation> implements Trainer<O> {

	protected Random random = new Random();

	protected Map<Integer, Hmm<O>> hmms;
	protected int numClasses; 
	protected int _stateCount; 
	protected int attributeValuesCount;
	protected int numAttributes;
	protected HMMHandler<O> handler;

	protected AbstractTrainer(int numClasses, int numAttributes, int stateCount,
			int attributeValuesCount) {
		this.numClasses = numClasses;
		this._stateCount = stateCount;
		this.attributeValuesCount = attributeValuesCount;
		this.numAttributes = numAttributes;	

	}
	
	public void setRandom(Random random) {
		this.random = random;
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
	
	public void setHMMHandler(HMMHandler<O> handler) {
		this.handler = handler;
	}

}
