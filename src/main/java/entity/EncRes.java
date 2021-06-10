package entity;

import org.nd4j.linalg.api.ndarray.INDArray;

public class EncRes {
    public INDArray W1Enc;
    public INDArray W2Enc;

    public INDArray Q1Enc;
    public INDArray Q2Enc;

    public EncRes(INDArray W1Enc, INDArray W2Enc, INDArray Q1Enc,INDArray Q2Enc){
        this.Q1Enc = Q1Enc;
        this.Q2Enc = Q2Enc;
        this.W1Enc = W1Enc;
        this.W2Enc = W2Enc;
    }
}
