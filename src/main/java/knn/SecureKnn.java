package knn;

import entity.EncRes;
import entity.Key;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import tool.CosCal;

import java.util.Arrays;
import java.util.Random;

public class SecureKnn {

    public Key key;

    public INDArray EncWord(INDArray d){
        d.muli(10);
        INDArray row = d.getRow(0);
        float[] word = row.toFloatVector();
        float[] wordExtend = Arrays.copyOf(word,word.length+2);

        //扩充向量
        wordExtend[word.length] = (float) Math.random();;
        wordExtend[word.length + 1] = 1;

        //分割
        float[] f1 = new float[wordExtend.length];
        float[] f2 = new float[wordExtend.length];
        float[] f3 = new float[wordExtend.length];
        float[] f4 = new float[wordExtend.length];

        for(int i = 0; i < wordExtend.length;i++){
            float random = new Random().nextFloat();
            if(key.S[i] == 1){
                f1[i] = wordExtend[i];
                f2[i] = wordExtend[i];
                f3[i] = 0.5f*wordExtend[i] + random;
                f4[i] = 0.5f*wordExtend[i] - random;
            }else {
                f1[i] = 0.5f*wordExtend[i] + random;
                f2[i] = 0.5f*wordExtend[i] - random;
                f3[i] = wordExtend[i];
                f4[i] = wordExtend[i];
            }
        }

        INDArray W1 = Nd4j.create(f1,new int[]{1,f1.length});
        INDArray W2 = Nd4j.create(f2,new int[]{1,f2.length});

        INDArray Q1 = Nd4j.create(f3,new int[]{f3.length,1});
        INDArray Q2 = Nd4j.create(f4,new int[]{f4.length,1});


        INDArray W1Ecn = W1.mmul(key.M1);

        INDArray W2Ecn = W2.mmul(key.M2);

        INDArray Q1Ecn = key.M1t.mmul(Q1);

        INDArray Q2Ecn = key.M2t.mmul(Q2);

        INDArray res = Nd4j.vstack(W1Ecn,W2Ecn,Q1Ecn.transpose(),Q2Ecn.transpose());

        return res;
    }

    public SecureKnn(Key key){
        this.key = key;
    }

    public static void main(String[] args) {
        SecureKnn secureKnn  = new SecureKnn(new KeyGen(3,5).GenerateKey());

        float[] w1 = {1f,2f,3f};
        float[] w2 = {1f,2f,3f};

        INDArray w1e = Nd4j.create(w1,new int[]{1, w1.length});
        INDArray w2e = Nd4j.create(w2,new int[]{1, w1.length});

        INDArray encRes = secureKnn.EncWord(w1e);
        INDArray encRes2 = secureKnn.EncWord(w2e);

        float cos = CosCal.CosCalculate2(encRes,encRes2);
        float cos2 = CosCal.CosCalculate(w1e,w2e);
        System.out.println(cos);
        System.out.println(cos2);
    }
}
