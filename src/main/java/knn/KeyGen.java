package knn;

import entity.Key;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.inverse.InvertMatrix;

import java.util.Arrays;

public class KeyGen {
    int d_plus;
    int d;

    INDArray CreatM(){
        INDArray m = Nd4j.rand(d_plus, d_plus);
        while(m.norm2Number().equals(0)){
            INDArray rand = Nd4j.eye(d_plus);
            m.add(rand);
        }
        return m;
    }

    int[] CreatS(){
        int[] S = new int[d_plus];
        for(int i = 0; i < d_plus; i++){
            double r = Math.random();
            S[i] = r > 0.5 ? 1 : 0;
        }
        return S;
    }

    float[] CreatW(){
        float[] W = new float[d_plus - d];
        for(int i = 0; i < d_plus - d; i++){
            W[i] = (float) Math.random();
        }
        return W;
    }

    public Key GenerateKey(){
        return new Key(CreatM(),CreatM(),CreatS(),CreatW());
    }

    public KeyGen(int d, int d_plus){
        this.d = d;
        this.d_plus = d_plus;
    }

    public static void main(String[] args) {
        KeyGen k = new KeyGen(1,3);
        Key key = k.GenerateKey();

        System.out.println(key.M1);
        INDArray Mt = InvertMatrix.invert(key.M1,false);
        System.out.println(Mt);
        Mt.mul(key.M1);
        System.out.println(Mt.mmul(key.M1));

        System.out.println(Arrays.toString(key.S));
        System.out.println(Arrays.toString(key.W));
    }
}
