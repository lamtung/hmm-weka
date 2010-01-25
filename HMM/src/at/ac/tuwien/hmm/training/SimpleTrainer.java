package at.ac.tuwien.hmm.training;

import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import weka.core.Instances;

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
public class SimpleTrainer<O extends Observation> extends AbstractTrainer<O> {
	public boolean DISPLAY = false;
	
	private int variations;
	
	/** for serialization */
	static final long serialVersionUID = -3481068294659183010L;

	
	public SimpleTrainer(int numClasses, int numAttributes, int stateCount,
			int attributeValuesCount, int variations) {
		super(numClasses, numAttributes,  stateCount,attributeValuesCount);
		this.variations = variations;
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

	public void trainHmms(Map<Integer, List<List<O>>> trainingInstancesMap, int accuracy,
			Instances data) throws Exception {
		
		double bestRatio = 0;

		// split the accuracy between init an final training
		// altogether we train variation times for init an one time for final
		// use the half of the time on init, rest on final training
		int accuracyPart = accuracy / (variations * 2);

		// initial training and comparing
		Map<Integer, Hmm<O>> bestHmms = null;
		for (int variation = 0; variation < variations; variation++) {
			initHmms();
			train(trainingInstancesMap, accuracyPart);
			handler.setHmms(getHmms());
			double ratio = handler.evaluate(data);
			if (ratio > bestRatio) {
				bestHmms = getHmms();
				bestRatio = ratio;
			}
			//System.out.println("Run " + variation + " " + ratio);
		}
		setHmms(bestHmms);

		// final round of training - use remaining iterations
		int remainingIterations = accuracy - (variations * accuracyPart);
		train(trainingInstancesMap, remainingIterations);
		handler.setHmms(getHmms());
	}
	
	
	private void train(Map<Integer, List<List<O>>> trainingInstancesMap, int accuracy) {
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



	
}
