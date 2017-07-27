package com.mark.wmd4j.demo;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;

import com.mark.wmd4j.WordMovers;
import com.mark.wmd4j.emd.EarthMovers;
import com.mark.word2vec.Word2VecUtils;

/** 
 * <p>Title: WMDDemo</p> 
 * <p>Description: </p>
 * @author Mark
 * @Date 2016-11-2
 */
public class WMDDemo {
	
	public static void main(String[] args) {
		WordVectors vectors = Word2VecUtils.loadWordVector(new File("your word2vec model path"));
		//默认
		WordMovers wm = WordMovers.builder().earthMovers(new EarthMovers()).wordVectors(vectors).build();
		
		//去除停用词
		/* WordMovers wm = WordMovers.builder().earthMovers(new EarthMovers())
				.wordVectors(vectors).stopwords(StopWords.getStopWords()).removeStopWords(true).build(); */
		
		//降低停用词的权值, stopwords 不为空时启用，默认为0.5，可设置
		/* WordMovers wm = WordMovers.builder().earthMovers(new EarthMovers())
				.wordVectors(vectors).stopwords(StopWords.getStopWords()).stopwordWeight(0.7).build(); */
		
		List<String> strings = new ArrayList<String>();
		for (int i = 0; i < 10; i++) {
			strings.add("I am very happy.");
		}
		wm.getSim(strings);
	}

}
