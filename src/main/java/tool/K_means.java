package tool;

import entity.ClusteringResults;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.*;
import java.util.stream.IntStream;

public class K_means {

    public static ClusteringResults k_means(Set<INDArray> set, int k, int it, int threadNum) {

        ArrayList<INDArray> docs = new ArrayList<>(set);

        if (docs.size() < k) {
            System.err.println("词数量小于聚类中心数量");
            System.exit(-1);
        }

        ArrayList<INDArray> res = new ArrayList<>();

        //随机选取k个矩阵作为中心
        int i = 0;
        while (i < k) {
            final double d = Math.random();
            final int index = (int) (d * docs.size());
            if (!res.contains(docs.get(index))) {
                res.add(docs.get(index));
                i++;
            }
        }


        Map<Integer, ArrayList<INDArray>> setHashMap = null;

        //迭代it次
        for (int t = 0; t < it; t++) {
            setHashMap = new HashMap<>();
            for (i = 0; i < k; i++) {
                setHashMap.put(i, new ArrayList<>());
            }

            Map<Integer, ArrayList<INDArray>> finalSetHashMap = setHashMap;
            Thread[] threads = new Thread[threadNum];
            int part = docs.size() / threadNum;

            //创建线程
            for (int threadIndex = 0; threadIndex < threadNum - 1; threadIndex++) {
                int finalThreadIndex = threadIndex;
                threads[threadIndex] = new Thread() {
                    @Override
                    public void run() {
                        CalThead(part * finalThreadIndex, part * (finalThreadIndex + 1), k, finalSetHashMap, docs, res);
                    }
                };
            }
            threads[threadNum - 1] = new Thread() {
                @Override
                public void run() {
                    CalThead(part * (threadNum - 1), docs.size(), k, finalSetHashMap, docs, res);
                }
            };

            //启动线程
            for (Thread th : threads) {
                try {
                    th.start();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
            for (Thread th : threads) {
                try {
                    //th.start();
                    th.join();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }

            setHashMap = new HashMap<>();
            i = 0;
            while (i < k) {
                setHashMap.put(i,new ArrayList<>());
                for(int j = 0; j <docs.size()/k; j++ ){
                    final double d = Math.random();
                    final int index = (int) (d * docs.size());
                    setHashMap.get(i).add(docs.get(index));
                }
                i++;
            }

            //更新中心点
            //遍历所有簇
            Thread[] threads2 = new Thread[k];
            int p = 0;
            for (ArrayList<INDArray> list : setHashMap.values()) {
                if (list.size() == 0) {
                    continue;
                }
                int finalP = p;
                threads2[p] = new Thread() {
                    @Override
                    public void run() {
                        float sum = 0;
                        int size = list.size();
                        //遍历所有矩阵的第i行，第j列
                        float[] total = new float[size];
                        Arrays.fill(total, 0);
                        float[][] temp = CosCal.CosCalculateSet(list);
                        for(int h = 0; h < temp.length; h++){
                            for(int j = 0; j < temp[0].length; j++){
                                total[h] += temp[h][j];
                            }
                        }
                        /*for(int m = 0; m < size;m++){
                            for(int n = m ; n < size;n++){
                                double res = CosCal.CosCalculate(finalList.get(m), finalList.get(n));
                                total[m] += res;
                                total[n] += res;
                            }
                        }*/
                        float max = Float.MAX_VALUE;
                        int index = 0;
                        for (int i = 0; i < size;i++){
                            if(total[i] < max){
                                index = i;
                                max = total[i];
                            }
                        }
                        //获取最小值下标
                        res.set(finalP, list.get(index));
                    }
                };
                //threads2[p].start();
                p++;
            }
            for (Thread thread : threads2) {
                try {
                    thread.start();
                } catch (Exception e) {
                    System.out.println("某一类为空，不影响程序运行，可忽略该条信息");
                }
            }

            for (Thread thread : threads2) {
                try {
                    //thread.start();
                    thread.join();
                } catch (Exception e) {
                    System.out.println("某一类为空，不影响程序运行，可忽略该条信息");
                }
            }

        }
        return new ClusteringResults(res, setHashMap);
    }

    public static ClusteringResults k_means2(Set<INDArray> set, int begin, int end, int k, int it, int threadNum) {

        ArrayList<INDArray> docs = new ArrayList<>(set);

        if (docs.size() < k) {
            System.err.println("词数量小于聚类中心数量");
            System.exit(-1);
        }

        ArrayList<INDArray> res = new ArrayList<>();

        //随机选取k个矩阵作为中心
        int i = 0;
        while (i < k) {
            final double d = Math.random();
            final int index = (int) (d * docs.size());
            if (!res.contains(docs.get(index))) {
                res.add(docs.get(index));
                i++;
            }
        }

        i = 0;
        Map<Integer, ArrayList<INDArray>> setHashMap = new HashMap<>();
        while (i < k) {
            setHashMap.put(i,new ArrayList<>());
            for(int j = 0; j <docs.size()/k; j++ ){
                final double d = Math.random();
                final int index = (int) (d * docs.size());
                setHashMap.get(i).add(docs.get(index));
            }
            i++;
        }

        return new ClusteringResults(res, setHashMap);
    }

    public static double[] k_cal(Set<INDArray> set,int threadNum) {

        int k = 0;
        ArrayList<INDArray> docs = new ArrayList<>(set);

        ArrayList<INDArray> res = new ArrayList<>();
        Map<Integer, ArrayList<INDArray>> setHashMap = null;
        double[] kScore = new double[500];

        for (k = 256; k < 257; k++) {
            System.out.println("第"+k+"个中心计算");

            //随机选取k个矩阵作为中心
            res = new ArrayList<>();
            //select(docs,res,k);
            int i = 0;
            while (i < k) {
                final double d = Math.random();
                final int index = (int) (d * docs.size());
                if (!res.contains(docs.get(index))) {
                    res.add(docs.get(index));
                    i++;
                }
            }

            setHashMap = new HashMap<>();
            for (i = 0; i < k; i++) {
                setHashMap.put(i, new ArrayList<>());
            }

            Map<Integer, ArrayList<INDArray>> finalSetHashMap = setHashMap;
            Thread[] threads = new Thread[threadNum];
            int part = docs.size() / threadNum;

            //创建线程
            for (int threadIndex = 0; threadIndex < threadNum - 1; threadIndex++) {
                int finalThreadIndex = threadIndex;
                int finalK1 = k;
                ArrayList<INDArray> finalRes = res;
                threads[threadIndex] = new Thread() {
                    @Override
                    public void run() {
                        CalThead(part * finalThreadIndex, part * (finalThreadIndex + 1), finalK1, finalSetHashMap, docs, finalRes);
                    }
                };
            }
            int finalK = k;
            ArrayList<INDArray> finalRes1 = res;
            threads[threadNum - 1] = new Thread() {
                @Override
                public void run() {
                    CalThead(part * (threadNum - 1), docs.size(), finalK, finalSetHashMap, docs, finalRes1);
                }
            };

            //启动线程
            for (Thread th : threads) {
                try {
                    th.start();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
            for (Thread th : threads) {
                try {
                    th.join();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
            System.out.println("开始累加");
            double sum = 0;
            for(int m = 0; m < k;m++){
                for (int n = 0; n < setHashMap.get(m).size();n++){
                    if(setHashMap.get(m).get(n) == null){
                        System.err.println(setHashMap.get(m).size() + "\t" + n);
                        continue;
                    }
                    sum+= CosCal.CosCalculate3(res.get(m), setHashMap.get(m).get(n));
                }
            }
            kScore[k-1] = sum;

        }
        return kScore;
    }

    static void select(ArrayList<INDArray> docs,ArrayList<INDArray> res,int k){
        final double d = Math.random();
        final int index = (int) (d * docs.size());
        res.add(docs.get(index));

        for(int i=0; i < (k-1); i++){
            int ind = 0;
            float[] sum = new float[docs.size()];

            for(INDArray doc:docs){
                for (INDArray r :res){
                    sum[ind] += CosCal.CosCalculate3(doc,r);
                }
                ind++;
            }

            int maxInd = 0;
            for(int t = 1; t < docs.size();t++){
                if (sum[maxInd] < sum[t]){
                    maxInd = t;
                }
            }

            res.add(docs.get(maxInd));
        }

    }

    static void CalThead(int begin, int end, int k, Map<Integer, ArrayList<INDArray>> finalSetHashMap, ArrayList<INDArray> docs,
                         ArrayList<INDArray> res) {
        float max = Float.MAX_VALUE;
        int index = 0;
        float score;
        for (int i = begin; i < end; i++) {
            int j;
            for (j = 0; j < k; j++) {
                score = CosCal.CosCalculate(docs.get(i), res.get(j));
                if (score < max) {
                    index = j;
                    max = score;
                }
            }
            //归类
            finalSetHashMap.get(index).add(docs.get(i));
            index = 0;
            max = Float.MAX_VALUE;
        }
    }
}
