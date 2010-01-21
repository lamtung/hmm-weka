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
import be.ac.ulg.montefiore.run.jahmm.Hmm;
import be.ac.ulg.montefiore.run.jahmm.ObservationInteger;
import be.ac.ulg.montefiore.run.jahmm.Opdf;
import be.ac.ulg.montefiore.run.jahmm.OpdfInteger;
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
	  
	@SuppressWarnings("unchecked")
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
	    
	    nominalsMap = new TreeMap<String, Integer>();
	    for (int attributeNo = 0; attributeNo < attributeCount;attributeNo++ ) {
	    	Attribute attribute = data.attribute(attributeNo);
	    	Enumeration attributeValues = attribute.enumerateValues();
	    	while (attributeValues.hasMoreElements()) {
		    	String value = (String)attributeValues.nextElement();
		    	if (!nominalsMap.containsKey(value)) {
		    		nominalsMap.put(value, nominalsMap.size());
		    	}
	    	}
	    }
	    
	    
	    hmms = new TreeMap<Integer, Hmm<ObservationInteger>>();
	    
	    Map<Integer, List<List<ObservationInteger>>> trainingInstancesMap = 
	    	new TreeMap<Integer, List<List<ObservationInteger>>>();
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
	    
	    Attribute classAttribute = 	data.classAttribute();
	    classAttribute.enumerateValues();
	    
	    
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

	    BaumWelchLearner learner = new BaumWelchLearner();
		learner.setNbIterations(accuracy); // "accuracy" - 
		//TODO: set in config
		
	    for (int classNo=0; classNo<numClasses; classNo++) {
	    	System.out.println("Training class "+classNo);
	    	List<List<ObservationInteger>> trainingInstances  = 
	    		trainingInstancesMap.get(classNo);
	    	Hmm<ObservationInteger> hmm = hmms.get(classNo);
	    	System.out.println("UnTrained HMM No "+classNo+":\r\n"+hmm.toString());
	    	Hmm<ObservationInteger> trainedHmm = hmm.clone();
	    	hmm = learner.learn(trainedHmm, trainingInstances);
	    	System.out.println("Trained HMM No "+classNo+":\r\n"+hmm.toString());
	    }
	    
	    System.out.println("building done");
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
