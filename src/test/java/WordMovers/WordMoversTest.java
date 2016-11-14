package WordMovers;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.junit.Test;

import com.mark.wmd4j.WordMovers;
import com.mark.wmd4j.emd.EarthMovers;
import com.mark.word2vec.Word2VecUtils;

/** 
 * <p>Title: WordMoversTest</p> 
 * <p>Description: </p>
 * @author Mark
 * @Date 下午3:18:11 2016年11月13日
 */
public class WordMoversTest {
	
	@Test
	public void test(){
		WordVectors vectors = Word2VecUtils.loadWordVector(new File("the wordvector's path"));
		WordMovers wm = WordMovers.builder().earthMovers(new EarthMovers()).wordVectors(vectors).build();
		List<String> strings = new ArrayList<String>();
		for (int i = 0; i < 10; i++) {
			strings.add("I am very happy.");
		}
		wm.getSim(strings);
	
	}

}
