package tool;

import com.google.gson.Gson;
import entity.ParaBean;

import java.io.BufferedReader;
import java.io.FileReader;

public class ParaReader {
    public static ParaBean ReadPara(String path){
        ParaBean paraBean = null;
        StringBuffer sbf = new StringBuffer();
        try{
            FileReader fr = new FileReader(path);
            BufferedReader bfr = new BufferedReader(fr);
            String temp;

            while ((temp = bfr.readLine()) != null){
                sbf.append(temp);
            }
            bfr.close();
            fr.close();
        }catch (Exception e){
            e.printStackTrace();
            System.exit(-1);
        }
        Gson gson = new Gson();
        paraBean = gson.fromJson(sbf.toString(),ParaBean.class);

        if(paraBean.docNumber == 0){
            paraBean.docNumber = Integer.MAX_VALUE;
        }
        if(paraBean.keywordNum == 0){
            paraBean.keywordNum = Integer.MAX_VALUE;
        }
        return paraBean;
    }
}
