package entity;

import org.nd4j.linalg.api.ndarray.INDArray;
import tool.CosCal;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.Map;

public class ClusteringResults {
    public ArrayList<INDArray> k_meas;
    public Map<Integer,ArrayList<INDArray>> setHashMap;

    public ClusteringResults(ArrayList<INDArray> k_meas, Map<Integer, ArrayList<INDArray>> setHashMap){
        this.k_meas = k_meas;
        this.setHashMap = setHashMap;
        Sort();
    }

    public void Sort(){
        for(int index: setHashMap.keySet()){
            ArrayList<INDArray> cl = setHashMap.get(index);
            INDArray center = k_meas.get(index);
            cl.sort(new Comparator<INDArray>() {
                @Override
                public int compare(INDArray o1, INDArray o2) {
                    if(CosCal.CosCalculate(o1,center) > CosCal.CosCalculate(o2,center)){
                        return -1;
                    }else
                        return 1;
                }
            });
        }
    }
}
