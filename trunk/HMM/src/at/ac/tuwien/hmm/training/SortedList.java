package at.ac.tuwien.hmm.training;


/**
 * Sorted List using Mergesort
 * Based on http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/merge/merge.htm
 * 
 * @author Christof Schmidt, H.W. Lang
 * @param <T>
 */
@SuppressWarnings("unchecked")
public class SortedList<T> {

	public SortedList(java.util.Collection<SortedEntry<T>> collection) {
		list = collection.toArray(list);
		sort();
	}

	private SortedEntry<T>[] list = new SortedEntry[0];
	private SortedEntry<T>[] temp;
	private int n;

	public void sort() {
		n = list.length;
		temp = new SortedEntry[(n + 1) / 2];
		mergesort(0, n - 1);
	}

	private void mergesort(int lo, int hi) {
		if (lo < hi) {
			int m = (lo + hi) / 2;
			mergesort(lo, m);
			mergesort(m + 1, hi);
			merge(lo, m, hi);
		}
	}

	void merge(int lo, int m, int hi) {
		int i, j, k;

		i = 0;
		j = lo;
		// vordere Hälfte von a in Hilfsarray b kopieren
		while (j <= m)
			temp[i++] = list[j++];

		i = 0;
		k = lo;
		// jeweils das nächstgrößte Element zurückkopieren
		while (k < j && j <= hi)
			if (temp[i].getValue() <= list[j].getValue())
				list[k++] = temp[i++];
			else
				list[k++] = list[j++];

		// Rest von b falls vorhanden zurückkopieren
		while (k < j)
			list[k++] = temp[i++];
	}
	
	public SortedEntry<T> getBottomEntry() {
		return list[0];
	}
	
	public SortedEntry<T> getTopEntry() {
		return list[list.length-1];
	}
	
	public SortedEntry<T> getEntry(int index) {
		return list[index];
	}
	
	public int size() {
		return n;
	}
}
