package at.ac.tuwien.hmm;

import java.text.NumberFormat;

import be.ac.ulg.montefiore.run.jahmm.Observation;

/**
 * This class implements observations whose values are taken out of a finite
 * set implemented as an enumeration.
 */
public class ObservationNominal extends Observation{

	public final String value;
	
	
	public ObservationNominal(String value)
	{
		this.value = value;
	}
	
	
	public String toString()
	{
		return value.toString();
	}
	
	
	public String toString(NumberFormat nf)
	{
		return toString();
	}
}

