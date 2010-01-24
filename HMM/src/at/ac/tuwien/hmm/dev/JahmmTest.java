package at.ac.tuwien.hmm.dev;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import at.ac.tuwien.hmm.HMMUtil;
import at.ac.tuwien.hmm.dev.HMMSetup.Action;
import be.ac.ulg.montefiore.run.jahmm.Hmm;
import be.ac.ulg.montefiore.run.jahmm.Observation;
import be.ac.ulg.montefiore.run.jahmm.ObservationDiscrete;
import be.ac.ulg.montefiore.run.jahmm.ObservationReal;
import be.ac.ulg.montefiore.run.jahmm.Opdf;
import be.ac.ulg.montefiore.run.jahmm.OpdfDiscrete;
import be.ac.ulg.montefiore.run.jahmm.learn.BaumWelchLearner;
import be.ac.ulg.montefiore.run.jahmm.toolbox.MarkovGenerator;

public class JahmmTest {
	Random random = new Random(0);
	/**
	 * Try whether the example from wikipedia can be replicated 
	 * http://en.wikipedia.org/wiki/Viterbi_algorithm#Example
	 */
	
	//public enum Action {W,S,C};
	
	public void runSimple() {
		
		double[] pi = {0.5, 0.5};
		HMMSetup hmmSetup = HMMSetup.getHMM(0);

		java.util.List<Opdf<ObservationDiscrete<HMMSetup.Action>>> opdfs = new ArrayList<Opdf<ObservationDiscrete<HMMSetup.Action>>>();
		for (double[] emission :hmmSetup.getEmissionMatrix()) {
			opdfs.add(new OpdfDiscrete<HMMSetup.Action>(HMMSetup.Action.class, emission) );
		}
		
		Hmm<ObservationDiscrete<HMMSetup.Action>> hmm = new Hmm<ObservationDiscrete<HMMSetup.Action>>(pi, hmmSetup.getTransitionMatrix(), opdfs);
		java.util.List<ObservationDiscrete<HMMSetup.Action>> oseq = new ArrayList<ObservationDiscrete<HMMSetup.Action>>();
		oseq.add(new ObservationDiscrete<HMMSetup.Action>(HMMSetup.Action.W));
		oseq.add(new ObservationDiscrete<HMMSetup.Action>(HMMSetup.Action.S));
		oseq.add(new ObservationDiscrete<HMMSetup.Action>(HMMSetup.Action.C));
		int[] result = hmm.mostLikelyStateSequence(oseq);
		for (int i:result) {
			System.out.print(i);
		}
		System.out.print("\t"+hmm.lnProbability(oseq));
	}

	/**
	 * Generates observation form a given HMM and trains randomly
	 * initialized HMM. 
	 * @throws Exception
	 */
	public void runTraining() {
		try {
			int observationSize = 60;
			
			// The original HMM to get the observations from and to compare the parameters
			HMMSetup hmmSetup = HMMSetup.getHMM(0);
			Hmm<ObservationDiscrete<HMMSetup.Action>> originalHmm = hmmSetup.getJaHMM();
			System.out.println("Original HMM:\r\n"+originalHmm.toString());
			
			// The training data
			List<List<ObservationDiscrete<HMMSetup.Action>>> observations = 
				getRandomObservations(originalHmm, observationSize, 60);
	
			BaumWelchLearner learner = new BaumWelchLearner();
			learner.setNbIterations(100); // "accuracy"
			
			for (int i=0; i<5; i++) {
				HMMSetup initialSetup = HMMSetup.getHMM(HMMSetup.INITIAL);
				Hmm<ObservationDiscrete<HMMSetup.Action>> initialHmm = initialSetup.getJaHMM();
		
				Hmm<ObservationDiscrete<HMMSetup.Action>> trainedHmm = learner.learn(initialHmm.clone(), observations);
				System.out.println("Trained HMM No "+i+":\r\n"+trainedHmm.toString());
			}		
			
			// for comparision, train a hmm using the original as the initial
			Hmm<ObservationDiscrete<HMMSetup.Action>> trainedHmm2 = learner.learn(originalHmm.clone(), observations);
			System.out.println("Trained HMM from original HMM:\r\n"+trainedHmm2.toString());
		} catch (CloneNotSupportedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
			
	}
	
	/**
	 * Trying to figure out what BW can and cannot do.
	 * @throws Exception
	 */
	public void runExtremeTraining() {
		try {
			int observationSize = 100;
			
			// The original HMM to get the observations from and to compare the parameters
			HMMSetup hmmSetup = HMMSetup.getHMM(0);
			Hmm<ObservationDiscrete<HMMSetup.Action>> originalHmm = hmmSetup.getJaHMM();
			System.out.println("Original HMM:\r\n"+originalHmm.toString());
			
			// The training data
			List<List<ObservationDiscrete<HMMSetup.Action>>> observations = 
				getRandomObservations(originalHmm, observationSize, 100);
	
			BaumWelchLearner learner = new BaumWelchLearner();
			learner.setNbIterations(100); // "accuracy"
			
			for (int i=0; i<1; i++) {
				HMMSetup initialSetup = HMMSetup.getHMM(HMMSetup.EXTREMETEST);
				Hmm<ObservationDiscrete<HMMSetup.Action>> initialHmm = initialSetup.getJaHMM();
				System.out.println("UnTrained HMM No "+i+":\r\n"+initialHmm.toString());
		
				Hmm<ObservationDiscrete<HMMSetup.Action>> trainedHmm = learner.learn(initialHmm.clone(), observations);
				System.out.println("Trained HMM No "+i+":\r\n"+trainedHmm.toString());
			}		
			
			// for comparision, train a hmm using the original as the initial
			Hmm<ObservationDiscrete<HMMSetup.Action>> trainedHmm2 = learner.learn(originalHmm.clone(), observations);
			System.out.println("Trained HMM from original HMM:\r\n"+trainedHmm2.toString());
		} catch (CloneNotSupportedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
			
	}
	
	private <O extends Observation> List<List<O>> getRandomObservations(Hmm<O> hmm, 
			int observationSize, int sequenceLength ) {
		MarkovGenerator<O> mg = new MarkovGenerator<O>(hmm);
		
		List<List<O>> observations = new ArrayList<List<O>>();		
		for (int i = 0; i < observationSize; i++) {
			observations.add(mg.observationSequence(sequenceLength));
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
			// The training data
			List<ObservationDiscrete<HMMSetup.Action>> emission= 
				getRandomObservations(hmm.getJaHMM(), 1, 60).get(0);
	
			
			
			for (int j=0; j<4;j++) {
				double prob = HMMSetup.getHMM(j).getJaHMM().lnProbability(emission);
				System.out.print((int)(prob*10)/10.0+"\t");
			}
			System.out.println();
		}
	}
	
	public void runTrainingAndClassificationTest()  {
		
	}

	public void runGenerateNominalDataSet()  {
		
		int observationSize = 100;
		int sequenceLength = 50;
	
		//generate sequences
		List<String> allObservations = new ArrayList<String>();
		for (int i=0; i<4; i++) {
			HMMSetup hmm = HMMSetup.getHMM(i);
			// The training data
			List<List<ObservationDiscrete<HMMSetup.Action>>> observations = 
				getRandomObservations(hmm.getJaHMM(), observationSize, sequenceLength);
			for (List<ObservationDiscrete<HMMSetup.Action>> observation: observations) {
				StringBuffer buffer = new StringBuffer();
				for (ObservationDiscrete<HMMSetup.Action> observationDiscrete: observation) {
					String s = null;
					switch (observationDiscrete.value) {
					case W:{s = "W"; break;}
					case S:{s = "S"; break;}
					case C:{s = "C"; break;}
					}
					buffer.append(s);
					buffer.append(",");
				}
				buffer.append("C"+i);
				allObservations.add(buffer.toString());
			}
		}
		
		// permute instances
		for (int i=0; i<allObservations.size()-1; i++) {
			int newPos = random.nextInt(allObservations.size()-i)+i;
			if (newPos != i) {
				String temp = allObservations.get(i);
				allObservations.set(i, allObservations.get(newPos));
				allObservations.set(newPos, temp);
			}
		}
		
		System.out.println("@relation nom_testhmm");

		for (int i=0; i<sequenceLength; i++) {
			String s = "000"+i;
			s=s.substring(s.length()-3);
			System.out.println("@attribute Seq_"+s+"  {W,S,C}");
		}
		System.out.println("@attribute Class {C0,C1,C2,C3}");
		System.out.println("");
		System.out.println("@data");


		for (String string: allObservations ) {
			System.out.println(string);
		}
			
		// OutPut
		
	}

	
	public void runGenerateNumericDataSet()  {
		
		int observationSize = 100;
		int sequenceLength = 50;
	
		//generate sequences
		List<String> allObservations = new ArrayList<String>();
		for (int i=0; i<4; i++) {
			Hmm<ObservationReal> hmm = HMMSetup.getNumericHMM(i);
			// The training data
			List<List<ObservationReal>> observations = 
				getRandomObservations(hmm, observationSize, sequenceLength);
			for (List<ObservationReal> observation: observations) {
				StringBuffer buffer = new StringBuffer();
				for (ObservationReal observationReal: observation) {
					String s = ""+Math.round(100*observationReal.value)/100.0;
					
					buffer.append(s);
					buffer.append(",");
				}
				buffer.append("C"+i);
				allObservations.add(buffer.toString());
			}
		}
		
		// permute instances
		for (int i=0; i<allObservations.size()-1; i++) {
			int newPos = random.nextInt(allObservations.size()-i)+i;
			if (newPos != i) {
				String temp = allObservations.get(i);
				allObservations.set(i, allObservations.get(newPos));
				allObservations.set(newPos, temp);
			}
		}
		
		System.out.println("@relation num_testhmm");

		for (int i=0; i<sequenceLength; i++) {
			String s = "000"+i;
			s=s.substring(s.length()-3);
			System.out.println("@attribute Seq_"+s+" numeric");
		}
		System.out.println("@attribute Class {C0,C1,C2,C3}");
		System.out.println("");
		System.out.println("@data");
		
		for (String string: allObservations ) {
			System.out.println(string);
		}
			
		// OutPut
		
	}

	public void runRandomTest() {
		int x[] = new int[10];
		for (int i=0; i<1000; i++) {
			int r = HMMUtil.getRandomStateCount(50, random);
			x[r]++;
		}
		
		for (int i=2; i<10; i++) {
			System.out.println(i+"\t" + x[i]/ 10.0);
		}
	}
	public static void main(String[] args) {
	/*	System.out.println("SIMPLE TEST:");
		new JahmmTest().runSimple();
		
		System.out.println("\r\n\r\nTRAINING TEST:");
		new JahmmTest().runTraining();
		
		System.out.println("\r\n\r\nCLASSIFY TEST:");
		new JahmmTest().runClassificationTest();
		 
*/		
		//System.out.println("\r\n\r\nGENERATING NOMINAL DATASETS:");
		//new JahmmTest().runGenerateNominalDataSet();

		/*System.out.println("\r\n\r\nGENERATING NUMERIC DATASETS:");
		*/new JahmmTest().runGenerateNumericDataSet();
		
//		System.out.println("\r\n\r\nGENERATING NUMERIC DATASETS:");
//		new JahmmTest().runExtremeTraining();
		
		//new JahmmTest().runRandomTest();
		
		
	}

}
