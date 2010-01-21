package at.ac.tuwien.hmm.training;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import at.ac.tuwien.hmm.HMMUtil;
import be.ac.ulg.montefiore.run.jahmm.Hmm;
import be.ac.ulg.montefiore.run.jahmm.ObservationInteger;
import be.ac.ulg.montefiore.run.jahmm.Opdf;
import be.ac.ulg.montefiore.run.jahmm.OpdfInteger;
import be.ac.ulg.montefiore.run.jahmm.learn.BaumWelchLearner;

public class SimpleTrainer implements Trainer {

	private Random random = new Random();
	private Map<Integer, Hmm<ObservationInteger>> hmms;
	private int numClasses; 
	private int stateCount; 
	private int attributeValuesCount;
	private int accuracy;
    BaumWelchLearner learner = new BaumWelchLearner();
	
	public SimpleTrainer(int numClasses, int stateCount,
			int attributeValuesCount, int accuracy) {
		this.numClasses = numClasses;
		this.stateCount = stateCount;
		this.attributeValuesCount = attributeValuesCount;
		this.accuracy = accuracy;
	}

	public void setRandom(Random random) {
		this.random = random;
	}
	
	public void initHmms() {
		hmms = new TreeMap<Integer, Hmm<ObservationInteger>>();
    	
	    for (int classNo=0;classNo<numClasses; classNo++ ) {
			double[][] transitionMatrix = HMMUtil.getRandomMatrix(stateCount, stateCount, random);
	    	double[][] emissionMatrix = HMMUtil.getRandomMatrix(stateCount, attributeValuesCount, random);
	    	java.util.List<Opdf<ObservationInteger>> opdfs = 
	    		new ArrayList<Opdf<ObservationInteger>>();
			for (double[] emission :emissionMatrix) {
				opdfs.add(new OpdfInteger(emission) );
			}
	    	Hmm<ObservationInteger> hmm = new Hmm<ObservationInteger>(
	    			HMMUtil.getRandomArray(stateCount, random), 
	    			transitionMatrix, opdfs);
	    	hmms.put(classNo, hmm);
	    }
    }

		
	public void trainHmms(Map<Integer, List<List<ObservationInteger>>> trainingInstancesMap) {
		initHmms();
		
		learner.setNbIterations(accuracy); // "accuracy" - 
		for (int classNo:trainingInstancesMap.keySet()) {
			List<List<ObservationInteger>> trainingInstances  = 
	    		trainingInstancesMap.get(classNo);
			Hmm<ObservationInteger> hmm = hmms.get(classNo);
	    	System.out.println("UnTrained HMM No "+classNo+":\r\n"+hmm.toString());
	    	Hmm<ObservationInteger> trainedHmm;
			try {
				trainedHmm = hmm.clone();
			} catch (CloneNotSupportedException e) {
				e.printStackTrace();
				trainedHmm = hmm;
			}
	    	hmm = learner.learn(trainedHmm, trainingInstances);
	    	System.out.println("Trained HMM No "+classNo+":\r\n"+hmm.toString());

		}		
	}

	public Map<Integer, Hmm<ObservationInteger>> getHmms() {
		return hmms;
	}
}
