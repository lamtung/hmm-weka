package at.ac.tuwien.hmm;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Vector;

public class HMMUtil {

	
	/**
	 * Create an array filled with the given value
	 */
	public static double[] getHomogenArray(int size, double value) {
		double[] d = new double[size];
		for (int i=0; i<size; i++) {
			d[i] = value;
		}
		return d;
	}
	
	/**
	 * Create a uniformly filled matrix (sums up to 1)
	 */
	public static double[][] getUniformMatrix(int rows, int columns) {
		double[][] matrix = new double[rows][];
		for(int i=0; i<rows;i++) {
			matrix [i] = getUniformArray(columns); 
		}
		return matrix;
	}
	
	/**
	 * Create a uniformly filled array (sums up to 1)
	 */
	public static double[] getUniformArray(int size) {
		double[] d = new double[size];
		double value = 1.0/size;
		for (int i=0; i<size; i++) {
			d[i] = value;
		}
		return d;
	}
	
	/**
	 * Create a Random filled matrix (sums up to 1)
	 */
	public static double[][] getRandomMatrix(int rows, int columns, Random random) {
		double[][] matrix = new double[rows][];
		for(int i=0; i<rows;i++) {
			matrix [i] = getRandomArray(columns, random); 
		}
		return matrix;
	}
	/**
	 * Create a Random filled array (sums up to 1)
	 */	
	public static double[] getRandomArray(int n, Random random) { //taken from dina
		double[] ps = new double[n];
		double sum = 0;
		// Generate random numbers
		for (int i=0; i<n; i++) {
			//prob should be larger than 0.00001
			while ((ps[i] = random.nextDouble()) <= 1.e-5) {};
			sum += ps[i];
		}
		// Scale to obtain a discrete probability distribution
		for (int i=0; i<n; i++) 
			ps[i] /= sum;
		return ps;
	}
	

	// generate a number between 2 and the log2 of of attributes. 
	// the probability of higher numbers decreases 
	public static int _getRandomStateCount(int numAttributes, Random random) {
		int maxStateCount = Math.max(2,(int)(Math.log(numAttributes) / Math.log(2)));
		int randomStateCount = Integer.MAX_VALUE;
		while (randomStateCount > maxStateCount) {
			randomStateCount = 1 + (int)(1.0 / (random.nextDouble() + 0.000000001));
		}
		
		return randomStateCount;
	}
	
	// the probability of higher numbers decreases 
	public static int getRandomStateCount(int numAttributes, Random random) {
		int maxStateCount = Math.max(2,(int)(Math.log(numAttributes) / Math.log(2)));
		while (true) {
			for (int i=2; i<= maxStateCount; i++) {
				if (random.nextDouble() < 0.34) {
					return i;
				}
			}
		}		
	}
	
	public static int newStates(Vector<Integer> tabuList, int numAttributes) {
		List<Integer> allStates = new ArrayList<Integer>();
		for (int i=2; i <= numAttributes; i++) {
			allStates.add(i);
		}
		for (Integer k : tabuList) {
			if (allStates.contains(k)) {
				allStates.remove(k);
			}
		}
		Collections.shuffle(allStates);
		return allStates.get(0);
		
	}
}
