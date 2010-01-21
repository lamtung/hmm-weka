package at.ac.tuwien.hmm;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Capabilities.Capability;
import at.ac.tuwien.hmm.training.SimpleTrainer;
import at.ac.tuwien.hmm.training.Trainer;
import be.ac.ulg.montefiore.run.jahmm.Hmm;
import be.ac.ulg.montefiore.run.jahmm.ObservationInteger;
import be.ac.ulg.montefiore.run.jahmm.learn.BaumWelchLearner;

public class HMMClassifier extends Classifier {
	
    private int numClasses;
    private Map<String, Integer> nominalsMap;
    private Map<Integer, Hmm<ObservationInteger>> hmms;
    private int attributeCount;
    private int accuracy = 50;
    private Random random;
    
	/** for serialization */
	static final long serialVersionUID = -3481068294659183989L;
	  
	public void buildClassifier(Instances data) throws Exception {
		
		random = data.getRandomNumberGenerator(0);
	    
		// can classifier handle the data?
	    getCapabilities().testWithFail(data);
	    
	    int stateCount = 2;// HACK - how to get this is unknown
	    int attributeValuesCount = 3; // HACK - get from attributes
	    attributeCount = data.numAttributes()-1;
	    // remove instances with missing class
	    data = new Instances(data);
	    data.deleteWithMissingClass();
	    
	    numClasses = data.numClasses();
	    
	    buildNominalsMap(data);
	    
	    Trainer trainer = new SimpleTrainer(numClasses, 
	    		stateCount, attributeValuesCount, accuracy);
	    
	    trainer.setRandom(random);
	    trainer.trainHmms(getTrainingInstances(data));
	    hmms = trainer.getHmms();
	    
	    System.out.println("building done");
	}
	
	@SuppressWarnings("unchecked")
	private void buildNominalsMap(Instances data) {
		nominalsMap = new TreeMap<String, Integer>();
	    for (int attributeNo = 0; attributeNo < attributeCount;attributeNo++ ) {
	    	Attribute attribute = data.attribute(attributeNo);
	    	Enumeration<String> attributeValues = 
	    		(Enumeration<String>)attribute.enumerateValues();
	    	while (attributeValues.hasMoreElements()) {
		    	String value = (String)attributeValues.nextElement();
		    	if (!nominalsMap.containsKey(value)) {
		    		nominalsMap.put(value, nominalsMap.size());
		    	}
	    	}
	    }
	}
	
	@SuppressWarnings("unchecked")
	private Map<Integer, List<List<ObservationInteger>>> getTrainingInstances(Instances data) {
	    Map<Integer, List<List<ObservationInteger>>> trainingInstancesMap = 
	    	new TreeMap<Integer, List<List<ObservationInteger>>>();
	        
		Enumeration<Instance> instances = (Enumeration<Instance>)data.enumerateInstances();
		
	    while (instances.hasMoreElements()) {
	    	Instance instance = instances.nextElement();
	    	int classNo = (int)instance.classValue();
	    	List<ObservationInteger> trainingObservation = 
	    		new ArrayList<ObservationInteger>();
	    	for (int attributeNo = 0; attributeNo < attributeCount;attributeNo++ ) {
	    		String attributeValue = instance.stringValue(attributeNo);
	    		int nominal = nominalsMap.get(attributeValue);
	    		ObservationInteger observation = 
	    			new ObservationInteger(nominal);
	    		trainingObservation.add(observation);
	    	}
	    	
	    	List<List<ObservationInteger>> trainingInstances = 
	    		trainingInstancesMap.get(classNo);
	    	if (trainingInstances == null) {
	    		trainingInstances = new ArrayList<List<ObservationInteger>>();
	    		trainingInstancesMap.put(classNo, trainingInstances);
	    	}
	    	trainingInstances.add(trainingObservation);
	    }
	    return trainingInstancesMap;
	}

	  /**
	   * Classifies the given test instance. The instance has to belong to a
	   * dataset when it's being classified. Note that a classifier MUST
	   * implement either this or distributionForInstance().
	   *
	   * @param instance the instance to be classified
	   * @return the predicted most likely class for the instance or 
	   * Instance.missingValue() if no prediction is made
	   * @exception Exception if an error occurred during the prediction
	   */
	  public double classifyInstance(Instance instance) throws Exception {
 		  List<ObservationInteger> observations = 
	    		new ArrayList<ObservationInteger>();
		  for (int attributeNo = 0; attributeNo < attributeCount;attributeNo++ ) {
				String attributeValue = instance.stringValue(attributeNo);
				int nominal = nominalsMap.get(attributeValue);
				ObservationInteger observation = 
					new ObservationInteger(nominal);
				observations.add(observation);
		  }
		  
		  int bestClass = -1;
		  double bestProbability = -100000;
		  
		  for (int classNo=0;classNo<numClasses; classNo++ ) {
			  Hmm<ObservationInteger> hmm = hmms.get(classNo);
			  double lnProbability = hmm.lnProbability(observations);
			  if (lnProbability > bestProbability) {
				  bestProbability = lnProbability;
				  bestClass = classNo;
			  }
		  }
		  
		  return bestClass;
	  }
	  
	  /**
	   * Returns a string describing classifier
	   * @return a description suitable for
	   * displaying in the explorer/experimenter gui
	   */
	  public String globalInfo() {
	    return "HMM Classifier.\n\n";
	
	  }
	  
	  public Capabilities getCapabilities() {
		    Capabilities result = super.getCapabilities();
		    result.disableAll();
		    
		    result.enable(Capability.NOMINAL_ATTRIBUTES);

		    // class
		    result.enable(Capability.NOMINAL_CLASS);

		    return result;
	  }
	  
	  /**
	   * Returns a description of this classifier.
	   *
	   * @return a description of this classifier as a string.
	   */
	  public String toString() {

	    return ("Test classifier");
	  }
	  
	  /**
	   * Main method for testing this class.
	   *
	   * @param argv the options
	   */
	  public static void main(String [] argv) {
	    runClassifier(new HMMClassifier(), argv);
	  }
}
