package at.ac.tuwien.hmm.training;

import java.util.List;
import java.util.Map;
import java.util.Random;

import be.ac.ulg.montefiore.run.jahmm.Hmm;
import be.ac.ulg.montefiore.run.jahmm.Observation;

/**
 * Superclass for all Training implementaions.
 * 
 * Subclass it to implement different strategies
 * 
 * @author Christof Schmidt
 *
 * @param <O> Type of Observation
 */
public interface Trainer<O extends Observation> extends java.io.Serializable {
	
	void setRandom(Random random);
	
	void initHmms();
	
	void perturbate1();
	
	void perturbate2();
	
	void trainHmms(Map<Integer, List<List<O>>> trainingInstancesMap, int accuracy);
	
	// Train HMM for a certain Class
	public Hmm<O> trainHmm(Map<Integer, List<List<O>>> trainingInstancesMap, int accuracy,int classNo);
	
	public Hmm<O> getHmm(int classNo);
	
	Map<Integer, Hmm<O>> getHmms();
	
	double[][] getNominalEmissionMatrix(int stateCount);

	double[] getNumericMeanArray(double givenMean,int stateCount);
	
	double[] getNumericVarianceArray(double givenVariance,int stateCount);
	
	void setHmms(Map<Integer, Hmm<O>> hmms);
	
}
