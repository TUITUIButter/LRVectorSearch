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

        // INDArray part = d.get(NDArrayIndex.all(),NDArrayIndex.interval(0,1,100));
        INDArray row = d.getRow(0);
        row.divi(row.norm2Number());

        float[] word = row.toFloatVector();
        float[] wordExtend = Arrays.copyOf(word,key.S.length);
        float[] queryExtend = Arrays.copyOf(word,key.S.length);

        float[] t = CreatT(key.S.length - word.length - 1);

        //扩充向量
        float n2 = row.norm2Number().floatValue();
        wordExtend[word.length] = (-0.5f * n2 * n2);
        queryExtend[word.length] = 1;

        System.arraycopy(key.W,0,wordExtend,word.length+1,key.W.length);
        System.arraycopy(t,0,queryExtend,word.length+1,t.length);

        //分割
        float[] f1 = new float[wordExtend.length];
        float[] f2 = new float[wordExtend.length];
        float[] f3 = new float[wordExtend.length];
        float[] f4 = new float[wordExtend.length];

        for(int i = 0; i < wordExtend.length;i++){
            float random = new Random().nextFloat();
            if(key.S[i] == 0){
                f1[i] = wordExtend[i];
                f2[i] = wordExtend[i];
                f3[i] = 0.5f*queryExtend[i] + random;
                f4[i] = 0.5f*queryExtend[i] - random;
            }else {
                f1[i] = 0.5f*wordExtend[i] + random;
                f2[i] = 0.5f*wordExtend[i] - random;
                f3[i] = queryExtend[i];
                f4[i] = queryExtend[i];
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

    float[] CreatT(int len){
        float[] T = new float[len];
        for(int i = 0; i < len - 1; i++){
            //T[i] = (float) Math.random();
            T[i] = key.W[i];
        }
        float sum = 0;
        for(int i = 0; i < len -1;i++){
            sum = sum + T[i] * key.W[i];
        }
        T[len - 1] = -sum/key.W[len - 1];
        return T;
    }

    public SecureKnn(Key key){
        this.key = key;
    }

    public static void main(String[] args) {
        SecureKnn secureKnn  = new SecureKnn(new KeyGen(3,6).GenerateKey());

        float[] w1 = {2f,2f,2f};
        float[] w2 = {1f,1f,1f};

        INDArray w1e = Nd4j.create(w1,new int[]{1, w1.length});
        INDArray w2e = Nd4j.create(w2,new int[]{1, w1.length});

        INDArray encRes = secureKnn.EncWord(w1e);
        INDArray encRes2 = secureKnn.EncWord(w2e);

        INDArray r1 = encRes.getRow(0).mmul(encRes2.getRow(2));
        INDArray r2 = encRes.getRow(1).mmul(encRes2.getRow(3));
        INDArray r = r1.add(r2);
        System.out.println(r);


    }
}
