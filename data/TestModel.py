import numpy as np
import joblib
import warnings
import ast

from views import preprocess

warnings.filterwarnings('ignore')


def test_model(param):
    my_prediction1=[]
    if(preprocess()=="valid"):

        model = open("Flight_RF.pkl", "rb")
        clfr = joblib.load(model)
        """
        parameters1 = [1.59,1737,0.67,219.61,1.3,1.64,185.86,233.84,239.13,7.08,7.03,7.28,1.49,0.47,0.9,6.51,2.15,0.51,-0.11,274,2637.73,1.22,2602,0.56,220.98,1.26,1.24,188.22,235.03,239.68,10.98,9.75,9.35,1.17,0.7,0.91,6.92,2.29,0.29,-0.11,254,1515.99,1.22,2602,0.56,220.98,1.26,1.24,188.22,235.03,239.68,10.98,9.75,9.35,1.17,0.7,0.91,6.92,2.29,0.29,-0.11,254,1515.99,1.16,3120,0.48,223.47,1.33,1.17,193.77,236.34,240.29,17.07,11.53,10.65,1.02,0.87,0.91,7.38,2.34,0.11,-0.1,262,848.44,1.16,3120,0.48,223.47,1.33,1.17,193.77,236.34,240.29,17.07,11.53,10.65,1.02,0.87,0.91,7.38,2.34,0.11,-0.1,262,848.44,1.16,3120,0.48,223.47,1.33,1.17,193.77,236.34,240.29,17.07,11.53,10.65,1.02,0.87,0.91,7.38,2.34,0.11,-0.1,262,848.44,1.16,3120,0.48,223.47,1.33,1.17,193.77,236.34,240.29,17.07,11.53,10.65,1.02,0.87,0.91,7.38,2.34,0.11,-0.1,262,848.44]
        #parameters2 = [12538,6,6,4,32768,1048576,5,958]
        #parameters2 = [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        inputFeature1 = np.asarray(parameters1).reshape(1, -1)
        #inputFeature2 = np.asarray(parameters2).reshape(1, -1)
        my_prediction1 = clfr.predict(inputFeature1)
        #my_prediction2 = clfr.predict(inputFeature2)
        print((my_prediction1[0]))
        #print(int(my_prediction2[0]))
    
        """
        float_list = ast.literal_eval(param)
        inputFeature1 = np.asarray(float_list).reshape(1, -1)
        my_prediction1 = clfr.predict(inputFeature1)
    else:
        my_prediction1[0]="NA"
    return my_prediction1[0]



#test_model()
#10407	9	6	6	33088	262144	4	952
#12538	6	6	4	32768	1048576	5	958

