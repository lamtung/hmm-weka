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
import be.ac.ulg.montefiore.run.jahmm.Hmm;
import be.ac.ulg.montefiore.run.jahmm.Observation;
import be.ac.ulg.montefiore.run.jahmm.ObservationInteger;
import be.ac.ulg.montefiore.run.jahmm.ObservationReal;
import be.ac.ulg.montefiore.run.jahmm.Opdf;
import be.ac.ulg.montefiore.run.jahmm.OpdfGaussian;
import be.ac.ulg.montefiore.run.jahmm.OpdfInteger;

public class HMMClassifier extends RandomizableClassifier {
	
    private Map<String, Integer> nominalsMap;
    private int numClasses;
    private int numAttributes;
    private int attributeCount;
    private Random random;
    private int attributeValuesCount;
    private HMMHandler<? extends Observation> handler;

    protected int m_Accuracy = 50;
    protected int m_States = -1;
    protected int m_Variations = 5; //TODO Wolfgang : make this configurable (default 1)
    								//it is the number of different initianl HMM setups to be 
    								// trained - the best setup will be chosen
        
	/** for serialization */
	static final long serialVersionUID = -3481068294659183000L;
	  
	public void buildClassifier(Instances data) throws Exception {
	    System.out.println("Start building");

	    
		random = data.getRandomNumberGenerator(getSeed());
	    
		// can classifier handle the data?
	    getCapabilities().testWithFail(data);
	    
	    attributeCount = data.numAttributes()-1;
	    // remove instances with missing class
	    data = new Instances(data);
	    data.deleteWithMissingClass();
	    
	    numClasses = data.numClasses();
	    numAttributes = data.numAttributes();

	    handler = getAttributeValueType(data);
	    
	    //train the HMMs
	    handler.train(data, m_Variations);
	    
	    System.out.println("building done");
	}
		
	private HMMHandler<? extends Observation> getAttributeValueType(Instances data)
			throws Exception {
		
		boolean isNominal = true;
		boolean isNumeric = true;
		
		for (int attributeNo = 0; attributeNo < attributeCount;attributeNo++ ) {
			Attribute attribute = data.attribute(attributeNo);
			if (!attribute.isNominal()) {
				isNominal = false;
			} 
			if (!attribute.isNumeric()) {
				isNumeric = false;
			} 
		}
		
		if (isNumeric) {
			if (isNominal) {
				throw new Exception("Failure: Both numeric and nominal values present!");
			} else {
				double[] allValues = new double[(data.numAttributes()-1) * data.numInstances()];
				for (int attributeNo = 0; attributeNo < attributeCount;attributeNo++ ) {
					double[] values = data.attributeToDoubleArray(attributeNo);
					System.arraycopy(values, 0, allValues, attributeNo * data.numInstances(), 
							data.numInstances());
				}
				final double mean = Utils.mean(allValues);
				final double variance = Utils.variance(allValues);
				
				return new HMMHandler<ObservationReal>(numClasses, numAttributes, m_States, 
						 attributeValuesCount, m_Accuracy, random) {
					/** for serialization */
					static final long serialVersionUID = -3481068294659183001L;

					public List<Opdf<ObservationReal>> createOdpf(int stateCount) {
						List<Opdf<ObservationReal>> opdfs = 
							new ArrayList<Opdf<ObservationReal>>();
						double[] means = getTrainer().getNumericMeanArray(mean, stateCount);
						double[] variances = getTrainer().getNumericVarianceArray(variance,stateCount);
						//HACK we should'n cast!
						for (int i = 0; i< means.length; i++) {
							opdfs.add(new OpdfGaussian(means[i],variances[i]));
						}
						return opdfs;
					}
			
					public ObservationReal createObservation(Instance instance, int attributeNo) {
						double value = instance.value(attributeNo);
						return new ObservationReal(value);
					}
				};			}
		} else {
			if (isNominal) {
			    //build an index over the nominal values
			    buildNominalsMap(data);

				return new HMMHandler<ObservationInteger>(numClasses, numAttributes, m_States, 
						 attributeValuesCount,  m_Accuracy, random) {
					/** for serialization */
					static final long serialVersionUID = -3481068294659183002L;

					public List<Opdf<ObservationInteger>> createOdpf(int stateCount) {
						List<Opdf<ObservationInteger>> opdfs = 
							new ArrayList<Opdf<ObservationInteger>>();
				    	for (double[] emission :getTrainer().getNominalEmissionMatrix(stateCount)) {
							opdfs.add(new OpdfInteger(emission) );
						}
						return opdfs;
					}
			
					public ObservationInteger createObservation(Instance instance, int attributeNo) {
						String value = instance.stringValue(attributeNo);
						int nominalId = nominalsMap.get(value);
						return new ObservationInteger(nominalId);
					}
				};
			} else {
				throw new Exception("Failure: Neither numeric nor nominal values present!");
			}		
		}
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
		    		int index = nominalsMap.size();
		    		nominalsMap.put(value, index);
		    	}
	    	}
	    }
	    this.attributeValuesCount = nominalsMap.keySet().size();
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
 		 
		int bestClass = this.handler.classifyInstance(instance);
		
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
		    result.enable(Capability.NUMERIC_ATTRIBUTES);

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
			    + "\t(default -1, this lets the algorithm choose the number)",
			      "N", -1, "-N <num>"));

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
	      setStates(-1);
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
