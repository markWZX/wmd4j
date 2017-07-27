package com.mark.wmd4j;

import java.util.List;
import java.util.Map;

import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.google.common.base.Preconditions;
import com.mark.util.nd4j.INDArrayUtil;

import lombok.Builder;
import lombok.NonNull;

/**
 * <p>
 * Title: CosWordSim
 * </p>
 * <p>
 * Description: 句子的余弦相似度
 * </p>
 * 
 * @author Mark
 * @Date 2016-11-30
 */
@Builder
public class WordCosSim {

	private static final double DEFAULT_STOPWORD_WEIGHT = 0.5;

	/**
	 * 词向量对象
	 */
	@NonNull
	private WordVectors wordVectors;

	/**
	 * 停用词
	 */
	private List<String> stopwords;

	/**
	 * 停用词权重
	 */
	private double stopwordWeight = DEFAULT_STOPWORD_WEIGHT;

	/**
	 * 分词工具
	 */
	private TokenPreProcess process;

	private String preProcess(String s) {
		Preconditions.checkNotNull(s, "String is null.");
		if (process == null) {
			process = new CommonPreprocessor();
		}
		return process.preProcess(s);
	}

	/**
	 * 余弦相似度
	 * 
	 * @param sents
	 * @return
	 */
	public double[][] getCosSim(List<String> sents) {
		double[][] cos = new double[sents.size()][sents.size()];
		for (int i = 0; i < sents.size() - 1; i++) {
			for (int j = i + 1; j < sents.size(); j++) {
				cos[i][j] = cosSim(sents.get(i), sents.get(j));
			}
		}
		for (int i = 0; i < cos.length; i++) {
			for (int j = 0; j <= i; j++) {
				if (i == j) {
					cos[i][j] = 1;
				} else {
					cos[i][j] = cos[j][i];
				}
			}
		}
		return cos;
	}

	/**
	 * 普通cos相似度
	 * 
	 * @param a
	 * @param b
	 * @return
	 */
	public double cosSim(String a, String b) {
		return cosSim(preProcess(a).split(" "), preProcess(b).split(" "), null);
	}

	/**
	 * 带有特殊词汇及其权重的余弦相似度
	 * 
	 * @param String
	 *            a
	 * @param String
	 *            b
	 * @param sepcialWords
	 * @return
	 */
	public double cosSim(String a, String b, Map<String, Double> sepcialWords) {
		return cosSim(preProcess(a).split(" "), preProcess(b).split(" "), sepcialWords);
	}

	/**
	 * 计算余弦相似度
	 * 
	 * @param tokensA
	 * @param tokensB
	 * @param sepcial
	 * @return
	 */
	private double cosSim(String[] tokensA, String[] tokensB, Map<String, Double> sepcial) {

		INDArray a = totalINDArray(tokensA, sepcial);
		INDArray b = totalINDArray(tokensB, sepcial);

		return INDArrayUtil.getCosSimilarity(a, b);
	}

	/**
	 * 合并 word embedding
	 * 
	 * @param tokens
	 * @return
	 */
	private INDArray totalINDArray(String[] tokens, Map<String, Double> sepcial) {
		INDArray total = Nd4j.zeros(wordVectors.lookupTable().layerSize());
		for (String token : tokens) {
			INDArray array = null;
			if (sepcial != null) {
				if (wordVectors.hasWord(token)) {
					array = wordVectors.getWordVectorMatrix(token)
							.mul(sepcial.get(token) == null ? 1 : sepcial.get(token));
				}
			} else {
				if (wordVectors.hasWord(token)) {
					array = wordVectors.getWordVectorMatrix(token);
				}
			}			
			if (array != null) {
				if (stopwords != null && stopwords.contains(token)) {
					array = array.mul(stopwordWeight);
				}
				total = total.add(array);			
			}

		}
		return total;
	}

}
