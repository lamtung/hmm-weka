package at.ac.tuwien.hmm.training;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;


import weka.core.Instances;
import at.ac.tuwien.hmm.HMMUtil;
import be.ac.ulg.montefiore.run.jahmm.Hmm;
import be.ac.ulg.montefiore.run.jahmm.Observation;
import be.ac.ulg.montefiore.run.jahmm.Opdf;
import be.ac.ulg.montefiore.run.jahmm.learn.BaumWelchLearner;
import be.ac.ulg.montefiore.run.jahmm.learn.BaumWelchScaledLearner;

/**
 * Training-with-multiple-initial-HMMs implementation.
 * Status: FAILED. 
 * Doesn't work this way, using the probality as a quality meter - 
 * the chosen HMMs have high probability for ALL instances, rendering
 * the classification process useless.
 * 
 * @author Christof Schmidt
 * 
 * @param <O>
 */
@SuppressWarnings("hiding")
public class MultiInitTrainer<O extends Observation> extends AbstractTrainer<O> {

	private Map<Integer, List<Hmm<O>>> hmmsList;	
	private int noInitials = 40;

	/** for serialization */
	static final long serialVersionUID = -3481068294659183020L;

	public MultiInitTrainer(int numClasses, int numAttributes, int stateCount,
			int attributeValuesCount) {
		super(numClasses, numAttributes,  stateCount,attributeValuesCount);
	}
	
	public void setRandom(Random random) {
		this.random = random;
	}

	public void initHmms() {
		
		hmmsList = new TreeMap<Integer, List<Hmm<O>>>();

		for (int classNo = 0; classNo < numClasses; classNo++) {
			List<Hmm<O>> hmmList = new ArrayList<Hmm<O>>();
			for (int i = 0; i < this.noInitials; i++) {
				int noOfStates = _stateCount; 
				if (noOfStates == -1) {
					noOfStates = HMMUtil.getRandomStateCount(numAttributes,random);
				}
				List<Opdf<O>> opdfs = handler.createOdpf(noOfStates);
				double[][] transitionMatrix = getMatrix(noOfStates, noOfStates,
						random);
				double[] pi = HMMUtil.getRandomArray(noOfStates, random);
				Hmm<O> hmm = new Hmm<O>(pi, transitionMatrix, opdfs);
				hmmList.add(hmm);
			}
			hmmsList.put(classNo, hmmList);
		}
	}

	public void trainHmms(Map<Integer, List<List<O>>> trainingInstancesMap, 
			int accuracy, Instances data) {
		initHmms();

		for (int classNo : trainingInstancesMap.keySet()) {
			List<List<O>> trainingInstances = trainingInstancesMap.get(classNo);
			List<List<O>> otherInstances = new ArrayList<List<O>>();
			for (int otherClassNo : trainingInstancesMap.keySet()) {
				if (classNo != otherClassNo) {
					List<List<O>> otherInstancesClass = trainingInstancesMap.get(otherClassNo);
					otherInstances.addAll(otherInstancesClass);
				}
			}
			List<Hmm<O>> initialHmms = hmmsList.get(classNo);
			trainHmm(trainingInstances, otherInstances, initialHmms, accuracy);
			hmms.put(classNo, initialHmms.get(0));
			System.out.println("Trained HMM No " + classNo + ":\r\n"
					+ initialHmms.get(0).toString());

		}
	}

	private void trainHmm(List<List<O>> trainingInstances,
			List<List<O>> otherInstances, List<Hmm<O>> hmms, int accuracy) {
		BaumWelchLearner learner = new BaumWelchScaledLearner();
		double stepSize = this.noInitials / (double) accuracy;
		double stepSum = 0;
		Map<Integer, Hmm<O>> idMap = new TreeMap<Integer, Hmm<O>>();
		for (int hmmNo = 0; hmmNo < hmms.size(); hmmNo++) {
			Hmm<O> hmm = hmms.get(hmmNo);
			idMap.put(hmmNo, hmm);
		}
		for (int i = 0; i < accuracy; i++) {

		
			for (int hmmNo = 0; hmmNo < hmms.size(); hmmNo++) {
				Hmm<O> hmm = hmms.get(hmmNo);
				Hmm<O> trainedHmm = learner.iterate(hmm, trainingInstances);
				hmms.set(hmmNo, trainedHmm);
				idMap.put(getKey(idMap, hmm), trainedHmm);
			}

			stepSum += stepSize;
			while (stepSum >= 1 && hmms.size() > 1) {
				int deleteCount = (int) stepSum;
				deleteCount = Math.min(deleteCount, hmms.size() - 1);
				stepSum = stepSum - deleteCount;
				
				
				SortedList<Hmm<O>> sortedList = getHmmSortedList(hmms,
						trainingInstances, idMap);
				SortedEntry<Hmm<O>> bestHmmEntry = sortedList.getTopEntry();

				System.out.println("Best HMM " + "\t" + bestHmmEntry.getValue()
						+ "\t" + hc(bestHmmEntry.getEntry(), idMap));
				if (!true) {
					SortedList<Hmm<O>> sortedOtherList = getHmmSortedList(hmms,
							otherInstances, idMap);
					SortedEntry<Hmm<O>> bestOtherHmmEntry = sortedOtherList.getTopEntry();
					System.out.println("Worst Other HMM"+"\t"+bestOtherHmmEntry.getValue()
							+"\t"+hc(bestOtherHmmEntry.getEntry(),idMap));

					for (int j=0; j<sortedList.size(); j++) {
						if (sortedList.getEntry(j).getEntry() == sortedOtherList.getEntry(j).getEntry()) {
							System.out.print("=");
						} else {
							System.out.print("!");
						}
							
					}
					System.out.println("");
					for (int delete = 0; delete < deleteCount/2; delete++) {
						SortedEntry<Hmm<O>> worstEntry = sortedList.getEntry(delete);
						hmms.remove(worstEntry.getEntry());
					}
					for (int delete = deleteCount/2; delete < deleteCount; delete++) {
						SortedEntry<Hmm<O>> bestOtherEntry = sortedOtherList.getEntry(
								sortedOtherList.size() -1 - (delete-deleteCount/2));
						hmms.remove(bestOtherEntry.getEntry());
					}
				} else {
					for (int delete = 0; delete < deleteCount/2; delete++) {
						SortedEntry<Hmm<O>> worstEntry = sortedList.getEntry(delete);
						hmms.remove(worstEntry.getEntry());
					}
					
				}
			}

		}
		System.out.println("FINAL HMM \t" + getKey(idMap, hmms.get(0))
				+ " size:" + hmms.size());

	}

	private int getKey(Map<Integer, Hmm<O>> idMap, Hmm<O> findHmm) {
		for (Integer key : idMap.keySet()) {
			Hmm<O> hmm = idMap.get(key);
			if (hmm == findHmm) {
				return key;
			}
		}
		return -1;
	}

	public SortedList<Hmm<O>> getHmmSortedList(List<Hmm<O>> hmms,
			List<List<O>> instances, Map<Integer, Hmm<O>> idMap) {
		List<SortedEntry<Hmm<O>>> list = new ArrayList<SortedEntry<Hmm<O>>>();

		for (int hmmNo = 0; hmmNo < hmms.size(); hmmNo++) {
			Hmm<O> hmm = hmms.get(hmmNo);
			double lnProbabilities = 0;
			for (List<O> instance : instances) {
				double lnProbability = hmm.lnProbability(instance);
				if (lnProbability > 0 ){
					int x = 1;
				}
				lnProbabilities += lnProbability;
			}
			SortedEntry<Hmm<O>> entry = new SortedEntry<Hmm<O>>(hmm,
					lnProbabilities);
			list.add(entry);
		}
		SortedList<Hmm<O>> sortedList = new SortedList<Hmm<O>>(list);
		return sortedList;
	}

	private int hc(Hmm<O> hmm, Map<Integer, Hmm<O>> idMap) {
		return getKey(idMap, hmm);
	}



}
