package com.mark.wmd4j;

import java.util.concurrent.atomic.AtomicInteger;

import lombok.AllArgsConstructor;
import lombok.Getter;

import org.nd4j.linalg.api.ndarray.INDArray;

@AllArgsConstructor
public class FrequencyVector {

	private volatile AtomicInteger frequency;
	@Getter private INDArray vector;

	public FrequencyVector(INDArray vector) {
		this(new AtomicInteger(1), vector);
	}

	public void incrementFrequency() {
		frequency.getAndIncrement();
	}

	public int getFrequency() {
		return frequency.get();
	}

}
