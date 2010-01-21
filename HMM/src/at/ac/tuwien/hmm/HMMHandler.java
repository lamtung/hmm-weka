package at.ac.tuwien.hmm;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import weka.core.Instance;
import weka.core.Instances;
import at.ac.tuwien.hmm.training.SimpleTrainer;
import at.ac.tuwien.hmm.training.Trainer;
import be.ac.ulg.montefiore.run.jahmm.Hmm;
import be.ac.ulg.montefiore.run.jahmm.Observation;
import be.ac.ulg.montefiore.run.jahmm.Opdf;

/**
 * This class needs to be implemented to get create the correct Opdf.
 * s
 * @author Christof Schmidt
 *
 * @param <O> Class of observation
 */
public abstract class HMMHandler<O extends Observation> {

	private Map<Integer, Hmm<O>> hmms;
	
	private Random random;
	private int numClasses; 
	private int stateCount; 
	private int numAttributes;
	private int accuracy;
	private Trainer<O> trainer;
	
	public HMMHandler(int numClasses, int stateCount,
			int numAttributes, int accuracy, Random random) {
		this.numClasses = numClasses;
		this.stateCount = stateCount;
		this.numAttributes = numAttributes;
		this.accuracy = accuracy;
		this.random = random;
		trainer = createTrainer();

	}

	
	// override this 
	public abstract List<Opdf<O>> createOdpf ();
	
	// override this 
	public abstract O createObservation (Instance instance, int attributeNo);
	
	public Map<Integer, List<List<O>>> getTrainingInstances(Instances data) {
	    Map<Integer, List<List<O>>> trainingInstancesMap = 
	    	new TreeMap<Integer, List<List<O>>>();
	        
		Enumeration<Instance> instances = (Enumeration<Instance>)data.enumerateInstances();
		
	    while (instances.hasMoreElements()) {
	    	Instance instance = instances.nextElement();
			int classNo = (int)instance.classValue();
	    	
			List<O> trainingObservation = 
				getObservationFromInstance(instance);
			
	    	List<List<O>> trainingInstances = 
	    		trainingInstancesMap.get(classNo);
	    	if (trainingInstances == null) {
	    		trainingInstances = new ArrayList<List<O>>();
	    		trainingInstancesMap.put(classNo, trainingInstances);
	    	}
	    	trainingInstances.add(trainingObservation);
	    }
	    return trainingInstancesMap;
	}
	
	/** Creates an Jahmm observation out of an WEKA instance
	 * 
	 * @param instance the instance to transform
	 * @return an observation list
	 */
	public List<O> getObservationFromInstance(Instance instance) {
    	List<O> trainingObservation = 
    		new ArrayList<O>();
    	for (int attributeNo = 0; attributeNo < instance.numAttributes()-1;attributeNo++ ) {
    		O observation = createObservation(instance, attributeNo);
    		trainingObservation.add(observation);
    	}
    	
    	return trainingObservation;
	}

	public void setHmms(Map<Integer, Hmm<O>> hmms) {
		this.hmms = hmms;
	}
	
	public void train(Instances data) {
	    trainer.setRandom(random);
	    trainer.trainHmms(getTrainingInstances(data));
	    setHmms(trainer.getHmms());
	}
	
	public Trainer<O> createTrainer() {
	    SimpleTrainer<O> trainer = 
	    	new SimpleTrainer<O>(numClasses, stateCount, numAttributes, accuracy, this);
	    return trainer;
	}
	
	
	  public int classifyInstance(Instance instance) throws Exception {
 		  List<O> observations = getObservationFromInstance(instance);

		  int bestClass = -1;
		  double bestProbability = -1000000000000.0;
		  
		  for (int classNo=0;classNo<numClasses; classNo++ ) {
			  Hmm<O> hmm = hmms.get(classNo);
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
}
