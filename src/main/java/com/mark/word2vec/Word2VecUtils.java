package com.mark.word2vec;

import java.io.File;
import java.io.IOException;

import lombok.extern.slf4j.Slf4j;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

/**
 * <p>
 * Title: Word2VecUtils
 * </p>
 * <p>
 * Description: Word2Vec's utils
 * </p>
 * 
 * @author Mark
 * @Date 下午3:31:56 2016-10-19
 */
@Slf4j
public class Word2VecUtils {

	/**
	 * 训练词向量
	 * @param sourceData
	 * @param output
	 * @throws Exception
	 */
	public static void buildAndWriteWordVec(File sourceData, File output)
			throws Exception {

		log.info("Load & Vectorize Sentences....");
		// Strip white space before and after for each line
		SentenceIterator iter = new BasicLineIterator(sourceData);
		// Split on white spaces in the line to get words
		TokenizerFactory t = new DefaultTokenizerFactory();
		t.setTokenPreProcessor(new CommonPreprocessor());

		log.info("Building model....");
		Word2Vec vec = new Word2Vec.Builder().minWordFrequency(5)
				.iterations(10).layerSize(100).seed(42).windowSize(5)
				.iterate(iter).tokenizerFactory(t).build();

		log.info("Fitting Word2Vec model....");

		vec.fit();
	
		log.info("Word2Vec model fitting finished....\n Start writing into file....");
		WordVectorSerializer.writeWord2Vec(vec, output);
		log.info("Write into file finished");
	}
	
	/**
	 * 增量训练模式，先读取原先的词向量，然后再在新的数据上做训练。
	 * @param sourceWord2vec
	 * @param newSourceData
	 * @param output
	 * @throws Exception
	 */
	public static void buildAndWriteWordVecWithUpdate(File sourceWord2vec, File newSourceData, File output)
			throws Exception {

		log.info("Load current Word2Vec model....");
		
		Word2Vec word2Vec = WordVectorSerializer.readWord2Vec(sourceWord2vec);
        /*
            PLEASE NOTE: after model is restored, it's still required to set SentenceIterator and TokenizerFactory, if you're going to train this model
         */
		log.info("Load current Word2Vec model....");
        SentenceIterator iterator = new BasicLineIterator(newSourceData);
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        word2Vec.setTokenizerFactory(tokenizerFactory);
        word2Vec.setSentenceIter(iterator);
        
        log.info("Word2vec uptraining...");

        word2Vec.fit();
		log.info("Word2Vec model uptraining finished....\n Start writing into file....");

		WordVectorSerializer.writeWord2Vec(word2Vec, output);
		
		log.info("Write into file finished");

	} 

	/**
	 * 新的词向量读取方法
	 * @param file
	 * @return
	 */
	public static WordVectors loadWordVector(File file) {
		WordVectors wordVectors = null;
		try {
			log.info("loading common dl4j's wordvectors");
			wordVectors = WordVectorSerializer.readWord2Vec(file);
		} catch (IOException e) {
			e.printStackTrace();
		}
		return wordVectors;
	}
	
	/**
	 * 旧的词向量读取方法
	 * @param file
	 * @return
	 */
	public static WordVectors loadTxtWordVector(File file) {
		WordVectors wordVectors = null;
		try {
			log.info("loading common dl4j's wordvectors");
			wordVectors = WordVectorSerializer.loadTxtVectors(file);
			log.info("common dl4j's wordvectors load complete");

		} catch (IOException e) {
			e.printStackTrace();
		}
		return wordVectors;
	}

	/**
	 * 读取Google词向量，是否为二进制模式
	 * @param file
	 * @param binary
	 * @return
	 */
	public static WordVectors loadGoogleWordVector(File file,boolean binary) {
		WordVectors wordVectors = null;
		try {
			log.info("loading google's wordvectors");
			wordVectors = WordVectorSerializer.loadGoogleModel(file, binary);
		} catch (IOException e) {
			e.printStackTrace();
		}
		return wordVectors;
	}
	
	public static WordVectors loadGoogleWordVector(File file) {
		return loadGoogleWordVector(file, true);
	}
}
