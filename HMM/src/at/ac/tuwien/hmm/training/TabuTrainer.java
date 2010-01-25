package at.ac.tuwien.hmm.training;

import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.Vector;


import weka.core.Instances;


import at.ac.tuwien.hmm.HMMUtil;
import be.ac.ulg.montefiore.run.jahmm.Hmm;
import be.ac.ulg.montefiore.run.jahmm.Observation;
import be.ac.ulg.montefiore.run.jahmm.Opdf;
import be.ac.ulg.montefiore.run.jahmm.learn.BaumWelchLearner;

/**
 * 
 * 
 * @author Lam Tung Nguyen
 *
 * @param <O>
 */
@SuppressWarnings("hiding")
public class TabuTrainer<O extends Observation> extends AbstractTrainer<O> {
	public boolean DISPLAY = false;
	

	/** for serialization */
	static final long serialVersionUID = -3481068294659183030L;

	private int iterationNumber;
	
	public TabuTrainer(int numClasses, int numAttributes, int stateCount,
			int attributeValuesCount, int iterationNumber) {
		super(numClasses, numAttributes,  stateCount,attributeValuesCount);
		this.iterationNumber = iterationNumber;
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
		initHmms();
		// Train the initial HMM with accuracy bw_accuracy
		trainHmms(trainingInstancesMap, accuracy);
		handler.setHmms(getHmms());

		// Train HMMs for each class
		for (int classNo = 0; classNo < data.numClasses(); classNo++) {
			assert (hmms.size() == data.numClasses());
			Vector<Integer> tabuList = new Vector<Integer>();

			Hmm<O> currentHmm = getHmm(classNo);
			Hmm<O> bestHmm = currentHmm;

			tabuList.add(currentHmm.nbStates());

			handler.setHmm(currentHmm, classNo);
			double currentRatio = handler.evaluate(data, classNo);
			double bestRatio = currentRatio;

			int k = 0;
			int pertube1Nb = 0;
			int pertube3Nb = 0;
			int pertubeNo = 1;
			for (int i = 1; i <= iterationNumber; i++) {

				// System.out.println("Tabu Search at Iteration " + i+
				// " of class " + classNo);
				if (k < 3) {
					if (pertube1Nb < 6) {
						perturbate1(classNo);
						pertube1Nb++;
						pertubeNo = 1;
					} else {
						pertube3Nb++;
						if (pertube3Nb == 5) {
							pertube1Nb = 0;
						}
						perturbate3(classNo);
						pertubeNo = 3;
					}

				} else {
					k = 0;
					int newStateNb = perturbate2(classNo, tabuList);
					tabuList.add(newStateNb);
				}
				Hmm<O> newHmm = trainHmm(trainingInstancesMap,
						accuracy, classNo);
				handler.setHmm(newHmm, classNo);
				double newRatio = handler.evaluate(data, classNo);

				// Tabu Criterion : New solution is acceptable if its cost is
				// higher than the old one otherwise remove
				if (newRatio <= currentRatio) {
					continue;
				} else {
					currentRatio = newRatio;
					currentHmm = newHmm;
					if (pertubeNo == 1)
						pertube1Nb = 0;
					if (pertube3Nb == 3)
						pertube3Nb = 0;

				}

				if (currentRatio > bestRatio) {
					// save the new best Hmms
					bestHmm = currentHmm;
					bestRatio = currentRatio;

					System.out.println("Best new ratio for class " + classNo
							+ " : " + bestRatio);
					System.out.println("Best HMM " + bestHmm.toString());
					if (bestRatio == 1.0) {
						break;
					}
					if (currentRatio - bestRatio < 0.01) {
						k = k + 1;
					}

				}

				handler.setHmm(bestHmm, classNo);
			}
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

	// Create a completely new HMM by changing the number of states. If the new number of states are 
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

	public Hmm<O> getHmm(int classNo) {
		return this.hmms.get(classNo);
	}


	
}
