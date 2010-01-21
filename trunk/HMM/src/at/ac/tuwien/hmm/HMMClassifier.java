package at.ac.tuwien.hmm;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.Vector;

import weka.classifiers.RandomizableClassifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import at.ac.tuwien.hmm.training.SimpleTrainer;
import be.ac.ulg.montefiore.run.jahmm.Hmm;
import be.ac.ulg.montefiore.run.jahmm.ObservationInteger;
import be.ac.ulg.montefiore.run.jahmm.Opdf;
import be.ac.ulg.montefiore.run.jahmm.OpdfInteger;

public class HMMClassifier extends RandomizableClassifier {
	
    private Map<String, Integer> nominalsMap;
    private Map<Integer, Hmm<ObservationInteger>> hmms;
    private int numClasses;
    private int attributeCount;
    private Random random;
    private int attributeValuesCount;
    
    protected int m_Accuracy = 50;
    protected int m_States = 2;
    
	/** for serialization */
	static final long serialVersionUID = -3481068294659183989L;
	  
	public void buildClassifier(Instances data) throws Exception {
		
		random = data.getRandomNumberGenerator(getSeed());
	    
		// can classifier handle the data?
	    getCapabilities().testWithFail(data);
	    
	    attributeCount = data.numAttributes()-1;
	    // remove instances with missing class
	    data = new Instances(data);
	    data.deleteWithMissingClass();
	    
	    numClasses = data.numClasses();
	    
	    //build an index over the nominal values
	    buildNominalsMap(data);

	    //train the HMMs
	    train(data);
	    
	    System.out.println("building done");
	}
	
	private void train(Instances data) {
	    OdpfCreator<ObservationInteger> odpfCreator = new OdpfCreator<ObservationInteger>() {
			public Opdf<ObservationInteger> createEmission(double[] emission) {
				return new OpdfInteger(emission);
			}
	    	
	    };
	    
	    SimpleTrainer<ObservationInteger> trainer = 
	    	new SimpleTrainer<ObservationInteger>(numClasses, 
	    	m_States, attributeValuesCount, m_Accuracy, odpfCreator);

	    trainer.setRandom(random);
	    trainer.trainHmms(getTrainingInstances(data));
	    hmms = trainer.getHmms();
        System.out.println(hmms.get(0).toString());

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
	    this.attributeValuesCount = nominalsMap.keySet().size();
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
				getObservationFromInstance(instance);
			
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
	
	/** Creates an Jahmm observation out of an WEKA instance
	 * 
	 * @param instance the instance to transform
	 * @return an observation list
	 */
	private List<ObservationInteger> getObservationFromInstance(Instance instance) {
    	List<ObservationInteger> trainingObservation = 
    		new ArrayList<ObservationInteger>();
    	for (int attributeNo = 0; attributeNo < attributeCount;attributeNo++ ) {
    		String attributeValue = instance.stringValue(attributeNo);
    		int nominal = nominalsMap.get(attributeValue);
    		ObservationInteger observation = 
    			new ObservationInteger(nominal);
    		trainingObservation.add(observation);
    	}
    	
    	return trainingObservation;
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
 			  getObservationFromInstance(instance);

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
	   * Returns an enumeration describing the available options.
	   *
	   * @return an enumeration of all the available options.
	   */
	  @SuppressWarnings("unchecked")
	public Enumeration listOptions() {

	    Vector newVector = new Vector(3);

	    newVector.addElement(new Option(
		      "\tAccuracy for Baum-Welch-Learner.\n"
		      + "\t(default 50)",
		      "A", 50, "-A <num>"));
	    newVector.addElement(new Option(
			      "\tNo of hidden states in the HMM.\n"
			      + "\t(default 2)",
			      "N", 2, "-N <num>"));

	    Enumeration enu = super.listOptions();
	    while (enu.hasMoreElements()) {
	      newVector.addElement(enu.nextElement());
	    }
	    return newVector.elements();
	  }

	  /**
	   * Parses a given list of options. Valid options are:<p>
	   *
	   * -A num <p>
	   * Sets the accuracy of the Baum-Welch-Learner <p>
	   * 
	   * *
	   * -N num <p>
	   * Sets the no of hidden states of the HMM <p>
	   * 
	   * Options after -- are passed to the designated classifier.<p>
	   *
	   * @param options the list of options as an array of strings
	   * @exception Exception if an option is not supported
	   */
	  public void setOptions(String[] options) throws Exception {
	    
	    String accuracy = Utils.getOption('A', options);
	    if (accuracy.length() != 0) {
	      setAccuracy(Integer.parseInt(accuracy));
	    } else {
	      setAccuracy(50);
	    }
	    String states = Utils.getOption('N', options);
	    if (states.length() != 0) {
	      setStates(Integer.parseInt(states));
	    } else {
	      setStates(2);
	    }

	    super.setOptions(options);
	  }

	  /**
	   * Gets the current settings of the classifier.
	   *
	   * @return an array of strings suitable for passing to setOptions
	   */
	  public String [] getOptions() {

	    String [] superOptions = super.getOptions();
	    String [] options = new String [superOptions.length + 4];

	    int current = 0;
	    options[current++] = "-A"; 
	    options[current++] = "" + getAccuracy();
	    options[current++] = "-N"; 
	    options[current++] = "" + getStates();

	    System.arraycopy(superOptions, 0, options, current, 
			     superOptions.length);

	    return options;
	  }
	  
	  /**
	   * Returns the tip text for this property
	   * @return tip text for this property suitable for
	   * displaying in the explorer/experimenter gui
	   */
	  public String accuracyTipText() {
	    return "The accuracy for the Baum-Welch-Learner.";
	  }

	  /**
	   * Set the accuracy for the Baum-Welch-Learner
	   *
	   * @param seed the accuracy 
	   */
	  public void setAccuracy(int accuracy) {

	    m_Accuracy = accuracy;
	  }

	  /**
	   * Gets the accuracy for the Baum-Welch-Learner
	   *
	   * @return the accuracy for the Baum-Welch-Learner
	   */
	  public int getAccuracy() {
	    
	    return m_Accuracy;
	  }
	  
	  /**
	   * Returns the tip text for this property
	   * @return tip text for this property suitable for
	   * displaying in the explorer/experimenter gui
	   */
	  public String statesTipText() {
	    return "The number of Hidden States in the HMM";
	  }

	  /**
	   * Sets the number of Hidden States in the HMM
	   *
	   * @param number of hidden states
	   */
	  public void setStates(int states) {

	    m_States = states;
	  }

	  /**
	   * Gets the number of states of the HMM
	   *
	   * @return the no of states of the HMM
	   */
	  public int getStates() {
	    
	    return m_States;
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
