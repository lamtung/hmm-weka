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
		super(numClasses, numAttributes, stateCount, attributeValuesCount);
		this.iterationNumber = iterationNumber;
	}

	public void initHmms() {

		hmms = new TreeMap<Integer, Hmm<O>>();
		for (int classNo = 0; classNo < numClasses; classNo++) {
			int noOfStates = _stateCount;
			if (noOfStates == -1) {
				noOfStates = HMMUtil.getRandomStateCount(numAttributes, random);
			}
			List<Opdf<O>> opdfs = handler.createOdpf(noOfStates);
			double[][] transitionMatrix = getMatrix(noOfStates, noOfStates,
					random);

			Hmm<O> hmm = new Hmm<O>(HMMUtil.getRandomArray(noOfStates, random),
					transitionMatrix, opdfs);
			hmms.put(classNo, hmm);
		}
	}
/*
	public void trainHmmsOld(Map<Integer, List<List<O>>> trainingInstancesMap,
			int accuracy, Instances data) throws Exception {

		System.out.println("Number of Iteration : " + iterationNumber);
		System.out.println("Baum Welcher Accuracy : " + accuracy);
		System.out.println("Number of states for generated model "
				+ _stateCount);
		// Initialize the trainer with start parameters
		initHmms();
		// Train the initial HMM with accuracy
		trainHmms(trainingInstancesMap, accuracy);
		handler.setHmms(getHmms());

		// Train HMMs for each class
		for (int classNo = 0; classNo < data.numClasses(); classNo++) {
			assert (hmms.size() == data.numClasses());
			// Vector<Integer> tabuList = new Vector<Integer>();

			Hmm<O> currentHmm = getHmm(classNo);
			Hmm<O> bestHmm = currentHmm;

			// tabuList.add(currentHmm.nbStates());

			handler.setHmm(currentHmm, classNo);
			double currentRatio = handler.evaluate(data);
			double bestRatio = currentRatio;

			int k = 0;
			int l = 0;
			// int pertube1Nb = 0;
			// int pertube3Nb = 0;
			// int pertubeNo = 1;

			int[] tabuList = new int[currentHmm.nbStates()];

			for (int i = 1; i <= iterationNumber; i++) {

				System.out.println("Tabu Search at Iteration " + i
						+ " of class " + classNo);
				if (l < 3 && k < 3) {
					//
					 // if (pertube1Nb < 6) { trainer.perturbate1(classNo);
					 // pertube1Nb++; pertubeNo = 1; } else { pertube3Nb++; if
					 // (pertube3Nb == 5) { pertube1Nb = 0; }
					 // trainer.perturbate3(classNo); pertubeNo = 3; }
					 //
					tabuList = perturbate4(classNo, tabuList);
					// perturbate3(classNo);

				} else {
					k = 0;
					l = 0;
					// int newStateNb = trainer.perturbate2(classNo, tabuList);

					// Randomize the model completely
					int nbStates = perturbate2(classNo);
					// tabuList.add(newStateNb);
					tabuList = new int[nbStates];
				}
				// Train the new HMM
				Hmm<O> newHmm = trainHmm(trainingInstancesMap, accuracy,
						classNo);
				handler.setHmm(newHmm, classNo);
				double newRatio = handler.evaluate(data);

				// Tabu Criterion : New solution is acceptable if its cost is
				// higher than the old one otherwise remove
				if (newRatio <= currentRatio) {
					l++;
					continue;
				} else {
					l = 0;
					currentRatio = newRatio;
					currentHmm = newHmm;
					
					 //if (pertubeNo == 1) pertube1Nb = 0; if (pertube3Nb == 3)
					 // pertube3Nb = 0;
					 
				}

				if (currentRatio > bestRatio) {
					// save the new best Hmms
					bestHmm = currentHmm;
					bestRatio = currentRatio;

					System.out.println("Best new ratio for class " + classNo
							+ " : " + bestRatio);
					System.out.println("Best HMM " + bestHmm.toString());

					if (currentRatio - bestRatio < 0.01) {
						k = k + 1;
					}

				} else {
					l = 3;
				}

				handler.setHmm(bestHmm, classNo);
			}
		}
	}
*/
	public void trainHmms(Map<Integer, List<List<O>>> trainingInstancesMap,
			int accuracy, Instances data) throws Exception {
		if (DISPLAY) {
			System.out.println("Number of Iteration : " + iterationNumber);
			System.out.println("Baum Welcher Accuracy : " + accuracy);
			System.out.println("Number of states for generated model "
					+ _stateCount);
		}
		// Initialize the trainer with start parameters
		initHmms();
		// Train the initial HMM with accuracy
		trainHmms(trainingInstancesMap, accuracy);
		handler.setHmms(hmms);

		assert (hmms.size() == data.numClasses());
		int[] tabuList = new int[data.numClasses()]; // Contains the frequency
														// of optimized Model

		// Map<Integer, Hmm<O>> currentHmms = getHmms();
		// Map<Integer, Hmm<O>> bestHmms = currentHmms; // Problem with
		// reference ? because when currentHmms change bestHms change also
		Map<Integer, Hmm<O>> bestHmms = new TreeMap<Integer, Hmm<O>>(hmms);
		double currentRatio = handler.evaluate(data);
		double bestRatio = currentRatio;

		int k = 0;
		for (int n = 1; n <= iterationNumber; n++) {
			if (DISPLAY) System.out.println("Tabu Search at Iteration " + n);
			// choose a HMM based on tabu list to optimize
			// Go through the tabu list and choose the first found least use row
			int bestFreq = 1000000;
			int classNo = -1; // Index of the Model that will be changed
			for (int i = 0; i < tabuList.length; i++) {
				if (tabuList[i] < bestFreq) {
					bestFreq = tabuList[i];
				}
			}

			for (int i = 0; i < tabuList.length; i++) {
				if (tabuList[i] == bestFreq) {
					classNo = i;
				}
			}

			if (DISPLAY) System.out.println("Optimizing class " + classNo);
			// tabuList[classNo]++;

			perturbate2(classNo);

			Hmm<O> newHmm = trainHmm(trainingInstancesMap, accuracy, classNo);
			handler.setHmm(newHmm, classNo);

			double newRatio = handler.evaluate(data);

			// Tabu Criterion : New solution is acceptable if its cost is
			// higher than the best one otherwise remove
			if (newRatio < bestRatio) {
				tabuList[classNo]++;
				continue;
			} else {

				tabuList[classNo] = 0; // reset frequency for ClassNo

				// save the new best Hmms
				bestHmms = new TreeMap<Integer, Hmm<O>>(hmms);
				bestRatio = newRatio;
				if (DISPLAY) System.out.println("Best new ratio found " + bestRatio);
				if (currentRatio - bestRatio < 0.01) {
					k = k + 1;
				}
				if (k == 3) {
					tabuList[classNo]++;
				}

			}

		}
		handler.setHmms(bestHmms);
	}

	public void trainHmms(Map<Integer, List<List<O>>> trainingInstancesMap,
			int accuracy) {
		BaumWelchLearner learner = new BaumWelchLearner();

		learner.setNbIterations(accuracy); // "accuracy" -
		for (int classNo : trainingInstancesMap.keySet()) {
			List<List<O>> trainingInstances = trainingInstancesMap.get(classNo);
			Hmm<O> hmm = hmms.get(classNo);
			if (DISPLAY)
				System.out.println("UnTrained HMM No " + classNo + ":\r\n"
						+ hmm.toString());
			Hmm<O> trainedHmm = learner.learn(hmm, trainingInstances);
			hmms.put(classNo, trainedHmm);
			if (DISPLAY)
				System.out.println("Trained HMM No " + classNo + ":\r\n"
						+ trainedHmm.toString());

		}
	}

	// Train HMM for a certain Class
	public Hmm<O> trainHmm(Map<Integer, List<List<O>>> trainingInstancesMap,
			int accuracy, int classNo) {
		BaumWelchLearner learner = new BaumWelchLearner();
		learner.setNbIterations(accuracy);
		List<List<O>> trainingInstances = trainingInstancesMap.get(classNo);
		Hmm<O> hmm = hmms.get(classNo);
		if (DISPLAY)
			System.out.println("UnTrained HMM No " + classNo + ":\r\n"
					+ hmm.toString());
		Hmm<O> trainedHmm = learner.learn(hmm, trainingInstances);
		// Update trained HMM
		hmms.put(classNo, trainedHmm);
		if (DISPLAY)
			System.out.println("Trained HMM No " + classNo + ":\r\n"
					+ trainedHmm.toString());

		return trainedHmm;

	}

	public Hmm<O> getHmm(int classNo) {
		return this.hmms.get(classNo);
	}

	// Create a completely new HMM by changing the number of states. If the new
	// number of states are
	public int perturbate2(int classNo) {
		assert (hmms != null);
		int newStateNb;
		// int newStateNb = HMMUtil.newStates(tabuList,numAttributes);
		if (_stateCount == -1) {
			newStateNb = HMMUtil.getRandomStateCount(numAttributes, random);
		} else {
			newStateNb = _stateCount;
		}

		List<Opdf<O>> opdfs = handler.createOdpf(newStateNb);
		double[][] transitionMatrix = getMatrix(newStateNb, newStateNb, random);
		Hmm<O> hmm = new Hmm<O>(HMMUtil.getRandomArray(newStateNb, random),
				transitionMatrix, opdfs);

		// update the new HMM
		hmms.put(classNo, hmm);
		return newStateNb;
	}

	public void perturbate1(int classNo) {
		assert (hmms != null);
		Hmm<O> hmm = hmms.get(classNo);
		int nbStates = hmm.nbStates();

		// Generate new pi array
		double[] newPi = HMMUtil.getRandomArray(nbStates, random);

		// Set the new pi array
		for (int i = 0; i < newPi.length; i++) {
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

	// Randomly change 1 row of the traistion matrix, use tabulist so that we
	// only change rows that were not so often changed before (use frequency)
	// The tabu list only exist within the small changes , after a big change
	// (change nbStates) it will be reset
	// tabuList contains the frequency of row that has been changed, e.g row 1
	// changed 5 time --> tabuList[1] == 5
	public int[] perturbate4(int classNo, int[] tabuList) {
		assert (hmms != null);
		Hmm<O> hmm = hmms.get(classNo);
		int nbStates = hmm.nbStates();
		double[] newTransRow = HMMUtil.getRandomArray(nbStates, random);

		// Go through the tabu list and choose the first found least use row
		int bestFreq = 1000000;
		int index = -1; // Index of row that will be changed
		for (int i = 0; i < tabuList.length; i++) {
			if (tabuList[i] < bestFreq) {
				bestFreq = tabuList[i];
			}
		}

		for (int i = 0; i < tabuList.length; i++) {
			if (tabuList[i] == bestFreq) {
				index = i;
			}
		}

		for (int i = 0; i < nbStates; i++) {
			hmm.setAij(index, i, newTransRow[i]);
		}

		tabuList[index]++;
		return tabuList;

	}

}
