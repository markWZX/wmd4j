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
		WordVectors vectors = Word2VecUtils.loadWordVector(new File("the wordvector's path"));
		WordMovers wm = WordMovers.builder().earthMovers(new EarthMovers()).wordVectors(vectors).build();
		List<String> strings = new ArrayList<String>();
		for (int i = 0; i < 10; i++) {
			strings.add("I am very happy.");
		}
		wm.getSim(strings);
	}

}
