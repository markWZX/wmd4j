package com.mark.wmd4j;

import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import lombok.Builder;
import lombok.Data;
import lombok.NonNull;

import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.google.common.base.Preconditions;
import com.mark.util.nd4j.INDArrayUtil;
import com.mark.wmd4j.emd.EarthMovers;

/**
 * Created by Majer on 21.9.2016.
 * 	<p>Refactored by
 * @author Mark on 11.3.2016.
 */
@Data
@Builder
public class WordMovers {

	private static final double DEFAULT_STOPWORD_WEIGHT = 0.5;
	/**
	 *  词向量对象
	 */
	@NonNull
	private WordVectors wordVectors;
	
	/**
	 * emd 算法
	 */
	@NonNull
	private EarthMovers earthMovers;
	
	/**
	 * 停用词
	 */
	private Set<String> stopwords;
	
	/**
	 * 停用词权重
	 */
	private double stopwordWeight = DEFAULT_STOPWORD_WEIGHT;
	
	/**
	 * 分词工具
	 */
	private TokenPreProcess process;

	
	/**
	 * 普通wmd距离
	 * @param a
	 * @param b
	 * @return
	 */
	public double distance(String a, String b) {
		return distance(preProcess(a).split(" "), preProcess(b).split(" "),null);
	}
	
	
	/**
	 * 带有特殊词汇及其权重的距离
	 * @param String a
	 * @param String b
	 * @param sepcialWords 
	 * @return
	 */
	public double distance(String a, String b, Map<String, Double> sepcialWords){
		return distance(preProcess(a).split(" "), preProcess(b).split(" "),sepcialWords);
	}

	private String preProcess(String s){
		Preconditions.checkNotNull(s,"String is null.");
		if (process == null) {
			process = new CommonPreprocessor();
		}
		return process.preProcess(s);
	}

	/**
	 * 计算两个句子的距离，带有特殊词
	 * @param tokensA
	 * @param tokensB
	 * @param sepcial
	 * @return
	 */
	private double distance(String[] tokensA, String[] tokensB, Map<String, Double> sepcial) {
		//Preconditions.checkArgument(tokensA.length < 1 || tokensB.length < 1,"tokens length should > 0");
		Map<String, FrequencyVector> mapA = bagOfVectors(tokensA);
		Map<String, FrequencyVector> mapB = bagOfVectors(tokensB);
		
		if (mapA.size() == 0 || mapB.size() == 0) {
			return Double.MAX_VALUE;
		}

		/*Preconditions.checkState(mapA.size() == 0 || mapB.size() == 0, 
				"Can't find any word vectors for given input text ..."
						+ Arrays.toString(tokensA) + "|"+ Arrays.toString(tokensB));*/

		// vocabulary of current tokens
		List<String> vocab = Stream.of(mapA.keySet(), mapB.keySet())
				.flatMap(Collection::stream).distinct()
				.collect(Collectors.toList());
		double matrix[][] = new double[vocab.size()][vocab.size()];

		for (int i = 0; i < matrix.length; i++) {
			String tokenA = vocab.get(i);
			for (int j = 0; j < matrix.length; j++) {
				String tokenB = vocab.get(j);
				if (mapA.containsKey(tokenA) && mapB.containsKey(tokenB)) {
					double distance = mapA.get(tokenA).getVector()
							.distance2(mapB.get(tokenB).getVector());
					// if tokenA and tokenB are stopwords, calculate distance according to stopword weight
					// the distance is cut half, just dicrease the effect of these words.
					if (stopwords != null && tokenA.length() != 1 && tokenB.length() != 1){
						distance *= stopwords.contains(tokenA) && stopwords.contains(tokenB) ? 1 : stopwordWeight;
					}
					matrix[i][j] = distance;
					matrix[j][i] = distance;
				}
			}
		}

		double[] freqA = wordFrequencies(vocab, mapA, sepcial);
		double[] freqB = wordFrequencies(vocab, mapB, sepcial);

		double dis = earthMovers.distance(freqA, freqB, matrix, 0);
		return dis;
	}

	/**
	 * 统计词频
	 * @param tokens
	 * @return
	 */
	private Map<String, FrequencyVector> bagOfVectors(String[] tokens) {

		Map<String, FrequencyVector> map = new LinkedHashMap<>(tokens.length);
		Arrays.stream(tokens)
				.filter(x -> wordVectors.hasWord(x))
				.forEach(
						x -> map.merge(
								x,
								new FrequencyVector(wordVectors
										.getWordVectorMatrix(x)), (v, o) -> {
									v.incrementFrequency();
									return v;
								}));

		return map;
	}
	
	private double[] wordFrequencies(List<String> vocab,
			Map<String, FrequencyVector> map,Map<String, Double> sepcial){
		if (sepcial == null) {
			return frequencies(vocab, map);
		}else {
			return frequencies(vocab, map, sepcial);
		}
	}
	
	/**
	 * Normalized frequencies for vocab and no sepcial words
	 * @param vocab
	 * @param map
	 * @return
	 */
	private double[] frequencies(List<String> vocab,
			Map<String, FrequencyVector> map){
		return vocab.stream().mapToDouble(x -> {
			if (map.containsKey(x)) {
				return (double) map.get(x).getFrequency() / map.size();
			}
			return 0d;
		}).toArray();
	}

	/*
	 * Normalized frequencies for vocab and add Weight to the special words
	 */
	private double[] frequencies(List<String> vocab,
			Map<String, FrequencyVector> map,Map<String, Double> sepcial) {
		return vocab.stream().mapToDouble(x -> {
			double dx = 0d;
			if (map.containsKey(x)) {
				dx = (double) map.get(x).getFrequency() / map.size();
			}
			if (sepcial.containsKey(x)) {
				dx *= sepcial.get(x);
			}
			return dx;
		}).toArray();
	}
	
	/**
	 * the matrix length = [sents.size()]*[sents.size()]
	 * contains specialWords
	 * @param sents
	 * @return
	 */
	public double[][] getWMDS(List<String> sents,Map<String, Double> sepcialWords) {
		double[][] wmd = new double[sents.size()][sents.size()];
		for (int i = 0; i < sents.size()-1; i++) {
			for (int j = i+1; j < sents.size(); j++) {
				wmd[i][j] = distance(sents.get(i), sents.get(j),sepcialWords);
			}
		}
		for (int i = 0; i < wmd.length; i++) {
			for (int j = 0; j <= i ; j++) {
				if (i == j) {
					wmd[i][j] = 0;
				}else {
					wmd[i][j] = wmd[j][i];
				}
			}
		}
		return wmd;

	}
	
	/**
	 * the matrix length = [sents.size()]*[sents.size()]
	 * contains specialWords 未归一化
	 * @param sents
	 * @return
	 */
	public double[][] getSim(List<String> sents,Map<String, Double> sepcialWords) {
		double[][] wmd = getWMDS(sents,sepcialWords);
		for (int i = 0; i < wmd.length; i++) {
			for (int j = 0; j < wmd[0].length; j++) {
				wmd[i][j] = 1 / (1 + wmd[i][j]);
			}
		}
		return wmd;
	}
	
	/**
	 * the matrix length = [sents.size()]*[sents.size()]
	 * contains specialWords 归一化
	 * @param sents
	 * @return
	 */
	public double[][] getNormSim(List<String> sents,Map<String, Double> sepcialWords) {
		double[][] sim = getSim(sents,sepcialWords);
		double max = 0d;
		for (int i = 0; i < sim.length; i++) {
			for (int j = i+1; j < sim[0].length; j++) {
				if (i!=j) {
					max = max > sim[i][j] ? max : sim[i][j];					
				}
			}
		}
		for (int i = 0; i < sim.length; i++) {
			for (int j = 0; j < sim[0].length; j++) {
				if (i!=j) {
					sim[i][j] /= max;					
				}
			}
		}
		return sim;
	}
	
	
	/**
	 * 3种不带特殊词的wmd用法
	 * 
	 */	
	public double[][] getWMDS(List<String> sents) {	
		return getWMDS(sents, null);
	}
	
	public double[][] getSim(List<String> sents) {
		return getSim(sents, null);
	}

	public double[][] getNormSim(List<String> sents) {
		return getNormSim(sents, null);
	}

	
	/**
	 * 余弦相似度
	 * @param sents
	 * @return
	 */
	public double[][] getCosSim(List<String> sents) {
		double[][] wmd = new double[sents.size()][sents.size()];
		for (int i = 0; i < sents.size()-1; i++) {
			for (int j = i+1; j < sents.size(); j++) {
				wmd[i][j] = cosSim(sents.get(i), sents.get(j));
			}
		}
		for (int i = 0; i < wmd.length; i++) {
			for (int j = 0; j <= i ; j++) {
				if (i == j) {
					wmd[i][j] = 1;
				}else {
					wmd[i][j] = wmd[j][i];
				}
			}
		}
		return wmd;
	}
	
	/**
	 * 普通cos相似度
	 * @param a
	 * @param b
	 * @return
	 */
	public double cosSim(String a, String b) {
		return cosSim(preProcess(a).split(" "), preProcess(b).split(" "),null);
	}
	
	
	/**
	 * 带有特殊词汇及其权重的余弦相似度
	 * @param String a
	 * @param String b
	 * @param sepcialWords 
	 * @return
	 */
	public double cosSim(String a, String b, Map<String, Double> sepcialWords){
		return cosSim(preProcess(a).split(" "), preProcess(b).split(" "),sepcialWords);
	}
	
	/**
	 * 计算余弦相似度
	 * @param tokensA
	 * @param tokensB
	 * @param sepcial
	 * @return
	 */
	private double cosSim(String[] tokensA, String[] tokensB, Map<String, Double> sepcial) {
		
		INDArray a = totalINDArray(tokensA,sepcial);
		INDArray b = totalINDArray(tokensB,sepcial);

		return INDArrayUtil.getCosSimilarity(a, b);
	}
	
	/**
	 * 合并 word embedding
	 * @param tokens
	 * @return
	 */
	private INDArray totalINDArray(String[] tokens,Map<String, Double> sepcial) {
		INDArray total = Nd4j.zeros(wordVectors.lookupTable().layerSize());
		for (String token : tokens) {
			if (sepcial != null) {
				if (wordVectors.hasWord(token)) {
					total = total.add(wordVectors.getWordVectorMatrix(token)
							.mul(sepcial.get(token) == null ? 1 : sepcial.get(token)));
				}
			}else {
				if (wordVectors.hasWord(token)) {
					total = total.add(wordVectors.getWordVectorMatrix(token));
				}
			}
			
		}
		return total;
	}
}
