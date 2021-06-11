package tool;

import entity.ClusteringResults;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.*;
import java.util.stream.IntStream;

public class K_means {
    public static ClusteringResults k_means(Set<INDArray> set, int begin, int end, int k, int it, int threadNum) {

        if (begin == -1) {
            begin = 0;
        }
        if (end == -1) {
            end = 300;
        }

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

            //更新中心点
            //遍历所有簇
            Thread[] threads2 = new Thread[k];
            int p = 0;
            for (ArrayList<INDArray> list : setHashMap.values()) {
                if (list.size() == 0) {
                    continue;
                }
                int finalP = p;
                ArrayList<INDArray> finalList = list;
                threads2[p] = new Thread() {
                    @Override
                    public void run() {
                        float sum = 0;
                        int size = finalList.size();
                        //遍历所有矩阵的第i行，第j列
                        float[] total = new float[size];
                        Arrays.fill(total, 0);
                        for(int m = 0; m < size;m++){
                            for(int n = m ; n < size;n++){
                                double res = CosCal.CosCalculate(finalList.get(m), finalList.get(n));
                                total[m] += res;
                                total[n] += res;
                            }
                        }
                        float max = Float.MAX_VALUE;
                        int index = 0;
                        for (int i = 0; i < size;i++){
                            if(total[i] < max){
                                index = i;
                                max = total[i];
                            }
                        }
                        //获取最小值下标
                        res.set(finalP, finalList.get(index));
                    }
                };
                //threads2[p].start();
                p++;
            }
            for (Thread thread : threads2) {
                try {
                    thread.start();
                } catch (Exception e) {
                    System.err.println("某一类为空，不影响程序运行，可忽略异常报错");
                }
            }

            for (Thread thread : threads2) {
                try {
                    //thread.start();
                    thread.join();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }

        }
        return new ClusteringResults(res, setHashMap);
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
