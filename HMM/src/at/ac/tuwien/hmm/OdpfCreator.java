package at.ac.tuwien.hmm;

import be.ac.ulg.montefiore.run.jahmm.Observation;
import be.ac.ulg.montefiore.run.jahmm.Opdf;

/**
 * This class needs to be implemented to get create the correct Opdf.
 * s
 * @author Christof Schmidt
 *
 * @param <O> Class of observation
 */
public abstract class OdpfCreator<O extends Observation> {

	public abstract Opdf<O> createEmission (double[] emission);
	
}
