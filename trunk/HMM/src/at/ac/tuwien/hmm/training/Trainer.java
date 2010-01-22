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
public interface Trainer<O extends Observation> {
	
	void setRandom(Random random);
	
	void trainHmms(Map<Integer, List<List<O>>> trainingInstancesMap);
	
	Map<Integer, Hmm<O>> getHmms();
	
	double[][] getNominalEmissionMatrix();

	double[] getNumericMeanArray(double givenMean);
	
	double[] getNumericVarianceArray(double givenVariance);
	
}
