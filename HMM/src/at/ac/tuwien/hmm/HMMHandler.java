package at.ac.tuwien.hmm;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.Vector;

import weka.core.Instance;
import weka.core.Instances;
import at.ac.tuwien.hmm.training.MultiInitTrainer;
import at.ac.tuwien.hmm.training.SimpleTrainer;
import at.ac.tuwien.hmm.training.Trainer;
import be.ac.ulg.montefiore.run.jahmm.Hmm;
import be.ac.ulg.montefiore.run.jahmm.Observation;
import be.ac.ulg.montefiore.run.jahmm.Opdf;

/**
 * This class needs to be implemented to get create the correct Opdf. s
 * 
 * @author Christof Schmidt
 * 
 * @param <O>
 *            Class of observation
 */
public abstract class HMMHandler<O extends Observation> implements
		java.io.Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private Map<Integer, Hmm<O>> hmms;

	private Random random;
	private int numClasses;
	private int stateCount;
	private int numAttributes;
	private int attributeValuesCount;
	private int accuracy;
	private Trainer<O> trainer;

	public HMMHandler(int numClasses, int numAttributes, int stateCount,
			int attributeValuesCount, int accuracy, Random random) {
		this.numClasses = numClasses;
		this.stateCount = stateCount;
		this.numAttributes = numAttributes;
		this.accuracy = accuracy;
		this.random = random;
		this.attributeValuesCount = attributeValuesCount;
		trainer = createTrainer();

	}

	// override this
	public abstract List<Opdf<O>> createOdpf(int stateCount);

	// override this
	public abstract O createObservation(Instance instance, int attributeNo);

	public Map<Integer, List<List<O>>> getTrainingInstances(Instances data) {
		Map<Integer, List<List<O>>> trainingInstancesMap = new TreeMap<Integer, List<List<O>>>();

		Enumeration<Instance> instances = (Enumeration<Instance>) data
				.enumerateInstances();

		while (instances.hasMoreElements()) {
			Instance instance = instances.nextElement();
			int classNo = (int) instance.classValue();
			// System.out.println(classNo+":"+instance.stringValue(instance.numAttributes()));
			List<O> trainingObservation = getObservationFromInstance(instance);

			List<List<O>> trainingInstances = trainingInstancesMap.get(classNo);
			if (trainingInstances == null) {
				trainingInstances = new ArrayList<List<O>>();
				trainingInstancesMap.put(classNo, trainingInstances);
			}
			trainingInstances.add(trainingObservation);
		}
		return trainingInstancesMap;
	}

	/**
	 * Creates an Jahmm observation out of an WEKA instance
	 * 
	 * @param instance
	 *            the instance to transform
	 * @return an observation list
	 */
	public List<O> getObservationFromInstance(Instance instance) {
		List<O> trainingObservation = new ArrayList<O>();
		for (int attributeNo = 0; attributeNo < instance.numAttributes() - 1; attributeNo++) {
			O observation = createObservation(instance, attributeNo);
			trainingObservation.add(observation);
		}

		return trainingObservation;
	}

	public void setHmms(Map<Integer, Hmm<O>> hmms) {
		this.hmms = hmms;
	}

	public double evaluate(Instances data) throws Exception {
		int correct = 0;
		for (int instanceNo = 0; instanceNo < data.numInstances(); instanceNo++) {
			Instance instance = data.instance(instanceNo);
			int bestClass = classifyInstance(instance);
			int classNo = (int) instance.classValue();
			if (bestClass == classNo) {
				correct++;
			}
		}
		return (double) correct / data.numInstances();
	}

	// Evaluate the precision for a certan classNo
	public double evaluate(Instances data, int myClassNo) throws Exception {
		int instanceOfClassCnt = 0;
		int correct = 0;
		for (int instanceNo = 0; instanceNo < data.numInstances(); instanceNo++) {
			Instance instance = data.instance(instanceNo);
			if ((int) instance.classValue() == myClassNo) {
				instanceOfClassCnt++;
				int bestClass = classifyInstance(instance);
				if (bestClass == myClassNo) {
					correct++;
				}
			}

		}
		return (double) correct / instanceOfClassCnt;
	}

	public void train(Instances data, int variations) throws Exception {
		trainer.setRandom(random);
		Map<Integer, List<List<O>>> trainingInstancesMap = getTrainingInstances(data);
		double bestRatio = 0;

		// split the accuracy between init an final training
		// altogether we train variation times for init an one time for final
		// use the half of the time on init, rest on final training
		int accuracyPart = accuracy / (variations * 2);

		// initial training and comparing
		Map<Integer, Hmm<O>> bestHmms = null;
		for (int variation = 0; variation < variations; variation++) {
			trainer.initHmms();
			trainer.trainHmms(trainingInstancesMap, accuracyPart);
			setHmms(trainer.getHmms());
			double ratio = evaluate(data);
			if (ratio > bestRatio) {
				bestHmms = trainer.getHmms();
				bestRatio = ratio;
			}
			System.out.println("Run " + variation + " " + ratio);
		}
		setHmms(bestHmms);

		// final round of training - use remaining iterations
		int remainingIterations = accuracy - (variations * accuracyPart);
		trainer.trainHmms(trainingInstancesMap, remainingIterations);
		setHmms(trainer.getHmms());
	}

	public void trainWithTabuSearch(Instances data, int iterationNumber)
			throws Exception {

		trainer.setRandom(random);
		Map<Integer, List<List<O>>> trainingInstancesMap = getTrainingInstances(data);
		// Initialize the trainer with start parameters
		trainer.initHmms();
		// Train the initial HMM with accuracy 10
		trainer.trainHmms(trainingInstancesMap, 10);
		setHmms(trainer.getHmms());

		// Train HMMs for each class
		for (int classNo = 0; classNo < data.numClasses(); classNo++) {
			assert (hmms.size() == data.numClasses());
			Vector<Integer> tabuList = new Vector<Integer>();
			double bestRatio = 0;			
			
			double currentRatio = 0;

			Hmm<O> currentHmm = trainer.getHmm(classNo);
			Hmm<O> bestHmm = currentHmm;
			
			tabuList.add(currentHmm.nbStates());

			//setHmm(currentHmm, classNo);
			System.out.println("Evaluating the initial HMM for class " + classNo);
			System.out.println("At the moment hmms has " + hmms.size() + " elements");
			System.out.println("All the keys are: ");
			System.out.println(hmms.keySet().toString());
			System.out.println("HMM for class 0 is ");
			System.out.println(hmms.get(0).toString());
			System.out.println("HMM for class 1 is ");
			System.out.println(hmms.get(1).toString());
			System.out.println("HMM for class 2 is ");
			System.out.println(hmms.get(2).toString());
			currentRatio = evaluate(data, classNo);

			int k = 0;
			boolean bestFound = true;
			for (int i = 1; i <= iterationNumber; i++) {
				System.out.println("Tabu Search: Begin iteration " + i
						+ " of class " + classNo);
				if (k < 3 && bestFound) {
					trainer.perturbate1(classNo);
				} else {
					k = 0;
					int newStateNb = trainer.perturbate2(classNo, tabuList);
					tabuList.add(newStateNb);
				}
				Hmm<O> newHmm = trainer.trainHmm(trainingInstancesMap, 10,
						classNo);				
				setHmm(newHmm, classNo);
				double newRatio = evaluate(data, classNo);

				// Tabu Criterion : New solution is acceptable if its cost is
				// higher than the old one otherwise remove
				if (newRatio <= currentRatio) {
					continue;
				} else {
					currentRatio = newRatio;
					currentHmm = newHmm;
				}

				if (currentRatio > bestRatio) {
					// save the new best Hmms
					bestHmm = currentHmm;
					bestFound = true;

					if (currentRatio - bestRatio < 0.01) {
						k = k + 1;
					}

				} else {
					bestFound = false;
				}

			}
			// DEBUG
			System.out.println("Setting the best found HMM for class "
					+ classNo);
			
			setHmm(bestHmm, classNo);			
		}

	}

	public void setHmm(Hmm<O> hmm, int classNo) {
		this.hmms.put(classNo, hmm);
	}

	public Trainer<O> createTrainer() {
		return new SimpleTrainer<O>(numClasses, numAttributes, stateCount,
				attributeValuesCount, this);
	}

	public Trainer<O> _createTrainer() {
		return new MultiInitTrainer<O>(numClasses, numAttributes, stateCount,
				attributeValuesCount, this);
	}

	public int classifyInstance(Instance instance) throws Exception {
		List<O> observations = getObservationFromInstance(instance);

		int bestClass = -1;
		double bestProbability = -1000000000000.0;

		for (int classNo = 0; classNo < numClasses; classNo++) {

			Hmm<O> hmm = this.hmms.get(classNo);
			// DEBUG
			if (hmm == null) {
				System.out.println("Can't find HMM for class " + classNo);
				for (int i = 0; i < 4; i++) {
					if (hmms.get(i) != null) {
						System.out.println("HMM number " + i);
					}
				}
				System.exit(1);
			} else {
				System.out.println("Found HMM for class " + classNo);
			}

			double lnProbability = hmm.lnProbability(observations);
			if (lnProbability > bestProbability) {
				bestProbability = lnProbability;
				bestClass = classNo;
			}
		}

		return bestClass;
	}

	public Trainer<O> getTrainer() {
		return trainer;
	}

	public Map<Integer, Hmm<O>> getHmms() {
		return hmms;
	}
}
