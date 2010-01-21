package at.ac.tuwien.hmm;

import java.util.ArrayList;
import java.util.Random;

import be.ac.ulg.montefiore.run.jahmm.ObservationInteger;

/**
 * Emitter for emitting HMM sequences. Like the markovgenerator in jahmm
 * @author Christof Schmidt
 *
 */
public class Emitter {

	private HMMSetup setup;
	private Random random;
	private int stateCount;
	
	public Emitter(HMMSetup setup, Random random) {
		super();
		this.setup = setup;
		this.random = random;
		stateCount = setup.stateNames.length;
	} 
	
	public String getRandomEmission(int size) {
		StringBuffer buffer = new StringBuffer();
		int state = random.nextInt(stateCount);// initial state
		for (int i=0; i<size; i++) {
			String emission = getEmission(state);
			buffer.append(emission);
			state = getNextState(state);
		}
		return buffer.toString();
	}
	
	private int getNextState(int oldState) {
		double randomValue = random.nextDouble();
		double[] probDist = setup.getTransitionMatrix()[oldState];
		int nextState = getIndex(randomValue, probDist);
		return nextState;
	}
	
	public java.util.List<ObservationInteger> getEmissionList(int size) {
		char[] randomEmission = this.getRandomEmission(size).toCharArray();
		java.util.List<ObservationInteger> emissionList = new ArrayList<ObservationInteger>();
		for (char c:randomEmission) {
			int index = setup.emissionSymbols.indexOf(c);
			emissionList.add(new ObservationInteger(index));
		}
		return emissionList;
	}
	
	private String getEmission(int state) {
		double randomValue = random.nextDouble();
		double[] probDist = setup.getEmissionMatrix()[state];
		int emissionIndex = getIndex(randomValue, probDist);
		return setup.getEmissionSymbols().substring(emissionIndex,emissionIndex+1);
	}
	
	private int getIndex(double randomValue, double[] probDist) {
		double cumProb = 0.0;
		int index = 0;
		while (randomValue >= cumProb) {
			cumProb += probDist[index++];
		}
		
		return index -1;
	}
	
}
