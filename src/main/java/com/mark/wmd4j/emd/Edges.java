package com.mark.wmd4j.emd;

import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;

@AllArgsConstructor
class Edge {
    
    int  to;
    long cost;
    
}

@AllArgsConstructor
class Edge0 {
	
    int  to;
    long cost;
    long flow;

}

@AllArgsConstructor
class Edge1 {
    
    int  to;
    long reducedCost;
    
}

@AllArgsConstructor
class Edge2 {
    
    int  to;
    long reducedCost;
    long residualCapacity;
    
}

@NoArgsConstructor
class Edge3 {
    
    int  to = 0;
    long dist = 0;

}