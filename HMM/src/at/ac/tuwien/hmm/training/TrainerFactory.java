package at.ac.tuwien.hmm.training;

import be.ac.ulg.montefiore.run.jahmm.Observation;

public class TrainerFactory<O extends Observation> {

	
	private int numClasses;
	private int stateCount;
	private int numAttributes;
	private int attributeValuesCount;
	private int variations;

	public TrainerFactory(int numClasses, int numAttributes, int stateCount,
			int attributeValuesCount, int variations) {
		this.numClasses = numClasses;
		this.stateCount = stateCount;
		this.numAttributes = numAttributes;
		this.variations = variations;
		this.attributeValuesCount = attributeValuesCount;
	}
	
	public Trainer<O> createTrainer(TrainerType trainerType) {
		switch (trainerType) {
			case Simple: {return createSimpleTrainer();}
			case MultiInit: {return createMultiInitTrainer();}
			case Tabu: {return createTabuTrainer();}
		}
		throw new RuntimeException(" No Trainer for"+ trainerType);
	}
	
	private Trainer<O> createSimpleTrainer() {
    	System.out.println("Method: Simple");
		return new SimpleTrainer<O>(numClasses, numAttributes, stateCount,
				attributeValuesCount, variations);
	}

	private Trainer<O> createMultiInitTrainer() {
		return new MultiInitTrainer<O>(numClasses, numAttributes, stateCount,
				attributeValuesCount);
	}
	
	private Trainer<O> createTabuTrainer() {
    	System.out.println("Method: Tabu");
    	return new TabuTrainer<O>(numClasses, numAttributes, stateCount,
				attributeValuesCount, variations);
	}
}
