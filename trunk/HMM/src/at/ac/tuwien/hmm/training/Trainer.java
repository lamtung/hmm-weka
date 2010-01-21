package at.ac.tuwien.hmm.training;

import java.util.List;
import java.util.Map;
import java.util.Random;

import be.ac.ulg.montefiore.run.jahmm.Hmm;
import be.ac.ulg.montefiore.run.jahmm.Observation;

public interface Trainer<O extends Observation> {
	
	void setRandom(Random random);
	
	void trainHmms(Map<Integer, List<List<O>>> trainingInstancesMap);
	
	Map<Integer, Hmm<O>> getHmms();
	
}
