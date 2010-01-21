package at.ac.tuwien.hmm;
import java.util.Random;

import dk.kvl.dina.Forward;
import dk.kvl.dina.HMM;
import dk.kvl.dina.SystemOut;
import dk.kvl.dina.Viterbi;

public class DinaTest {
	
	/**
	 * Generates observation form a given HMM and trains randomly
	 * initialized HMM. 
	 */
	public void runTrainingTest()  {
		Random random = new Random(0);
		int classes = 4;
		int obsLength = 40;
		int obsCount = 50;
		
		// train the HMMs (no initial hmm needed here
		HMM[] trainedHmms = new HMM[classes];
		for (int j=0; j<classes; j++) {
			System.out.println("Training Hmm #"+j);
			HMMSetup hmmSetup = HMMSetup.getHMM(j);
			String[] observations = getObservations(hmmSetup, random, obsLength, obsCount);

			HMM estimate = HMM.baumwelch(observations, hmmSetup.getStateNames(), 
					hmmSetup.getEmissionSymbols(), 0.0001);
			trainedHmms[j] = estimate;
		}
		
		// print out logProbabilities for different hmms/emissions
		System.out.println("\r\nCheck Probabilities");
		for (int j=0; j<classes; j++) {
			System.out.println("\r\nTesting emission form HMM #"+j);
			System.out.println("HMM\tPr Tra\tPr orig");
			Emitter emitter = new Emitter(HMMSetup.getHMM(j), random);
			String emission = emitter.getRandomEmission(obsLength);
			
			for (int i=0; i<classes; i++) {
				Forward forwardE = new Forward(trainedHmms[i], emission);
				Forward forwardR = new Forward(HMMSetup.getHMM(i).getDinaHMM(), emission);
				System.out.print(i+"\t"+(int)(forwardE.logprob()*10)/10.0+"\t");
				System.out.println((int)(forwardR.logprob()*10)/10.0+"\t");
			}

			System.out.println();
			
		}
		for (int i=0; i<classes; i++) {
			System.out.println("\r\n\r\n Trained HMM #"+i);
			trainedHmms[i].print(new SystemOut());
			System.out.println("\r\n Original HMM #"+i);
			HMMSetup.getHMM(i).getDinaHMM().print(new SystemOut());
		}
	}
	
	private String[] getObservations(HMMSetup hmmSetup, Random random, int obsLength, int obsCount) {
		String[]observations = new String[obsCount];
		Emitter emitter = new Emitter(hmmSetup, random);
		for (int i=0;i<observations.length; i++) {
			observations[i] = emitter.getRandomEmission(obsLength);
			//System.out.println(observations[i]);
		}
		return observations;
	}
	
	/**
	 * Classification Test. Generate emissions from different HMMs. 
	 * Check each emission against all hmms - the one which has generated
	 * it should have the highest probability. (the highest logvalue)
	 * HMM setups again from wikipedia example.
	 */
	public void runClassificationTest()  {
		for (int i=0; i<4; i++) {
			
			HMMSetup hmm = HMMSetup.getHMM(i);
			Emitter emitter = new Emitter(hmm, new Random(0));
			String emission = emitter.getRandomEmission(100);
			
			System.out.print(emission+"\t");
			
			
			for (int j=0; j<4;j++) {
				Forward forward = new Forward(HMMSetup.getHMM(j).getDinaHMM(), emission);
				System.out.print((int)(forward.logprob()*10)/10.0+"\t");
			}
			System.out.println();
		}

	}
	
	/**
	 * Try whether the example from wikipedia can be replicated 
	 * http://en.wikipedia.org/wiki/Viterbi_algorithm#Example
	 */

	public void runSimpleTest() {
		HMM hmm = HMMSetup.getHMM(0).getDinaHMM();
		String observations = "WSC";
		Viterbi viterbi = new Viterbi(hmm, observations);
		System.out.println("State-Path: " + viterbi.getPath());

		Forward forward = new Forward(hmm, observations);
		System.out.println("Probability of path: "+
				Math.pow(Math.E, forward.logprob()));
		
	}
	
	public static void main(String[] args) {
		System.out.println("SIMPLE TEST:");
		new DinaTest().runSimpleTest();
		
		System.out.println("\r\n\r\nCASSIFICATION TEST:");
		new DinaTest().runClassificationTest();
		
		System.out.println("\r\n\r\nTRAINING TEST:");
		new DinaTest().runTrainingTest();
	}
	
	


	

}
