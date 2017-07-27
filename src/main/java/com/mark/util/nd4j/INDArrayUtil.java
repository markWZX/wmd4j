package com.mark.util.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * 
 * <p>Title: INDArrayUtil</p> 
 * <p>Description:  only for one dimension,INDArray的各种计算
 * @author Mark
 * @Date 下午2:52:03 2016年7月1日
 */
public class INDArrayUtil {
	
	/**
	 * 	模
	 * @param a
	 * @return
	 */
    public static double getINDArrayNorm(INDArray a){
    	return Math.sqrt(transvection(a,a));
    } 
    
    /**
     * 内积
     * @param a
     * @param b
     * @return
     */
    public static double transvection(INDArray a, INDArray b){
    	return Nd4j.getBlasWrapper().dot(a, b);
    }
    
    /**
     * 去中心化
     * @param a
     * @return
     */
    public static INDArray decentration(INDArray a){
    	INDArray mean = a.mean(1);
    	INDArray array = a.sub(mean.getDouble(0));
    	return array;
    }
        
    /**
     * 余弦相似度
     * @param a
     * @param b
     * @return
     */
    public static double getCosSimilarity(INDArray a, INDArray b){
    	return Transforms.cosineSim(a, b);    	
    }
    
}
