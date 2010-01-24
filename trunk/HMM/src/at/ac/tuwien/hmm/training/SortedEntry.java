package at.ac.tuwien.hmm.training;

public class SortedEntry<T> {
	
	private double value;
	private T entry;
	
	public SortedEntry(T entry, double value) {
		this.value = value;
		this.entry = entry;
	}

	public double getValue() {
		return value;
	}

	public T getEntry() {
		return entry;
	}
	
}
