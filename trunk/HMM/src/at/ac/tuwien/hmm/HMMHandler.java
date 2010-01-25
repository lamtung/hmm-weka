package at.ac.tuwien.hmm;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import weka.core.Instance;
import weka.core.Instances;
import at.ac.tuwien.hmm.training.Trainer;
import at.ac.tuwien.hmm.training.TrainerFactory;
import at.ac.tuwien.hmm.training.TrainerType;
import be.ac.ulg.montefiore.run.jahmm.Hmm;
import be.ac.ulg.montefiore.run.jahmm.Observation;
import be.ac.ulg.montefiore.run.jahmm.Opdf;

/**
 * This class needs to be implemented to get create the correct Opdf. s
 * 
 * 
 * @author Christof Schmidt
 * 
 * @param <O>
 *            Class of observation
 */
public abstract class HMMHandler<O extends Observation> implements
		java.io.Serializable {

	private boolean IS_EVALUATE_ALL = false; //switch evaluating against all instances on/off
	
	private static final long serialVersionUID = 1L;

	private Map<Integer, Hmm<O>> hmms;

	private Random random;
	private int numClasses;
	private int accuracy;
	private Trainer<O> trainer;

	public HMMHandler(int numClasses, int numAttributes, int stateCount,
			int attributeValuesCount, int accuracy, int variations, 
			TrainerType trainerType, Random random) {
		this.numClasses = numClasses;

		this.accuracy = accuracy;
		this.random = random;
		trainer = new TrainerFactory<O>(numClasses, numAttributes, stateCount,
				attributeValuesCount, variations).createTrainer(trainerType);
		trainer.setHMMHandler(this);

	}

	// implement this
	public abstract List<Opdf<O>> createOdpf(int stateCount);

	// implement this
	public abstract O createObservation(Instance instance, int attributeNo);

	@SuppressWarnings("unchecked")
	public Map<Integer, List<List<O>>> getTrainingInstances(Instances data) {
		Map<Integer, List<List<O>>> trainingInstancesMap = new TreeMap<Integer, List<List<O>>>();

		Enumeration<Instance> instances = (Enumeration<Instance>) data
				.enumerateInstances();

		while (instances.hasMoreElements()) {
			Instance instance = instances.nextElement();
			int classNo = (int) instance.classValue();
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

	public double evaluate(Instances data, int myClassNo) throws Exception {
		if (IS_EVALUATE_ALL) {
			return evaluate(data);
		} else {
			return _evaluate(data, myClassNo);
		}
	}
	
	// Evaluate the precision for a certan classNo
	public double _evaluate(Instances data, int myClassNo) throws Exception {
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
		trainer.trainHmms(trainingInstancesMap, accuracy, data);
	}


	public void setHmm(Hmm<O> hmm, int classNo) {
		this.hmms.put(classNo, hmm);
	}

	public int classifyInstance(Instance instance) throws Exception {
		List<O> observations = getObservationFromInstance(instance);

		int bestClass = -1;
		double bestProbability = -1000000000000.0;

		for (int classNo = 0; classNo < numClasses; classNo++) {

			Hmm<O> hmm = this.hmms.get(classNo);

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

	public void setHmms(Map<Integer, Hmm<O>> hmms) {
		this.hmms = hmms;
	}

}
