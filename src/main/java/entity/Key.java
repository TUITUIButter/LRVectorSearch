package entity;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.inverse.InvertMatrix;

public class Key {
    public INDArray M1;
    public INDArray M2;
    public INDArray M1t;
    public INDArray M2t;
    public int[] S;
    public float[] W;

    public Key(INDArray M1,INDArray M2,int[] S,float[] W){
        this.M1 = M1;
        this.M2 = M2;
        this.S = S;
        this.W = W;
        this.M1t = InvertMatrix.invert(M1,false);
        this.M2t = InvertMatrix.invert(M2,false);
    }
}
