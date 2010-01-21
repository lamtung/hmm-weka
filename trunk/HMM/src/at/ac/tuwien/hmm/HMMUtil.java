package at.ac.tuwien.hmm;

import java.util.Random;

public class HMMUtil {

	
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
}
