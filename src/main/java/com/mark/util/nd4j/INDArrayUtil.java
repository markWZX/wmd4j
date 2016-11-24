package com.mark.util.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * 
 * <p>Title: INDArrayUtil</p> 
 * <p>Description:  only for one dimension,INDArray's compute
 * @author Mark
 * @Date 2016-7-1
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

    	double result = 0;
    	for (int i = 0; i < a.length(); i++) {
			result+=a.getDouble(i)*b.getDouble(i);
		}

    	return result;
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
    	
    	double result = 0;  	
    	if (getINDArrayNorm(a) != 0 && getINDArrayNorm(b) != 0) {
    		
			result =  transvection(a, b)/(getINDArrayNorm(a)*getINDArrayNorm(b));
		} 	
    	return result;
    	
    }

}
