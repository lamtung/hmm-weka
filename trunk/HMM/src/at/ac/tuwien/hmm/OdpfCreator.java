package at.ac.tuwien.hmm;

import be.ac.ulg.montefiore.run.jahmm.Observation;
import be.ac.ulg.montefiore.run.jahmm.Opdf;


public abstract class OdpfCreator<O extends Observation> {

	public abstract Opdf<O> createEmission (double[] emission);
	
}
