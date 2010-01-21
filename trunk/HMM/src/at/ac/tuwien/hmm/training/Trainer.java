package at.ac.tuwien.hmm.training;

import java.util.List;
import java.util.Map;
import java.util.Random;

import be.ac.ulg.montefiore.run.jahmm.Hmm;
import be.ac.ulg.montefiore.run.jahmm.ObservationInteger;

public interface Trainer {
	
	void setRandom(Random random);
	
	void trainHmms(Map<Integer, List<List<ObservationInteger>>> trainingInstancesMap);
	
	Map<Integer, Hmm<ObservationInteger>> getHmms();
	
}
