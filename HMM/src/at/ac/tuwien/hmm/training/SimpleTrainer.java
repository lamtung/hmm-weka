package at.ac.tuwien.hmm.training;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import at.ac.tuwien.hmm.HMMUtil;
import at.ac.tuwien.hmm.OdpfCreator;
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
public class SimpleTrainer<O extends Observation> implements Trainer<O> {

	private Random random = new Random();
	private Map<Integer, Hmm<O>> hmms;
	private int numClasses; 
	private int stateCount; 
	private int attributeValuesCount;
	private int accuracy;
	private OdpfCreator<O> odpfCreator;
    BaumWelchLearner learner = new BaumWelchLearner();
	
	public SimpleTrainer(int numClasses, int stateCount,
			int attributeValuesCount, int accuracy,
			OdpfCreator<O> odpfCreator) {
		this.numClasses = numClasses;
		this.stateCount = stateCount;
		this.attributeValuesCount = attributeValuesCount;
		this.accuracy = accuracy;
		this.odpfCreator = odpfCreator;
	}

	public void setRandom(Random random) {
		this.random = random;
	}
	
	public void initHmms() {
		
		hmms = new TreeMap<Integer, Hmm<O>>();
    	
	    for (int classNo=0;classNo<numClasses; classNo++ ) {
			double[][] transitionMatrix = HMMUtil.getRandomMatrix(stateCount, stateCount, random);
	    	double[][] emissionMatrix = HMMUtil.getRandomMatrix(stateCount, attributeValuesCount, random);
	    	java.util.List<Opdf<O>> opdfs = 
	    		new ArrayList<Opdf<O>>();
			for (double[] emission :emissionMatrix) {
				opdfs.add(odpfCreator.createEmission(emission) );
			}
	    	Hmm<O> hmm = new Hmm<O>(
	    			HMMUtil.getRandomArray(stateCount, random), 
	    			transitionMatrix, opdfs);
	    	hmms.put(classNo, hmm);
	    }
    }

		
	public void trainHmms(Map<Integer, List<List<O>>> trainingInstancesMap) {
		initHmms();
		
		learner.setNbIterations(accuracy); // "accuracy" - 
		for (int classNo:trainingInstancesMap.keySet()) {
			List<List<O>> trainingInstances  = 
	    		trainingInstancesMap.get(classNo);
			Hmm<O> hmm = hmms.get(classNo);
	    	System.out.println("UnTrained HMM No "+classNo+":\r\n"+hmm.toString());
    	    Hmm<O> trainedHmm = learner.learn(hmm, trainingInstances);
	     	hmms.put(classNo, trainedHmm);
	    	System.out.println("Trained HMM No "+classNo+":\r\n"+trainedHmm.toString());

		}		
	}

	public Map<Integer, Hmm<O>> getHmms() {
		return hmms;
	}
}
