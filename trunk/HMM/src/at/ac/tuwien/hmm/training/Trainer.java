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
	
	void trainHmms(Map<Integer, List<List<O>>> trainingInstancesMap);
	
	Map<Integer, Hmm<O>> getHmms();
	
	double[][] getNominalEmissionMatrix(int stateCount);

	double[] getNumericMeanArray(double givenMean,int stateCount);
	
	double[] getNumericVarianceArray(double givenVariance,int stateCount);
	
}
