'''
Created on 16 Jun 2017

@author: martin
'''
import _mysql as ms
import numpy as np
import sys
filename='data.npz'
def get_table(db,name,count=None,block=1000):
    if count is None:
        db.query("SELECT count(*) from %s"%name)
        rs=db.store_result()
        count=int(rs.fetch_row()[0][0])
    ret=np.zeros(shape=(count,28*28+1),dtype=np.uint8)
    done=False
    offset=0
    index=0
    while not done:
        db.query("SELECT * from %s limit %i,%i"%(name,offset,block))
        rs=db.store_result()
        r=rs.fetch_row()
        while r:
            data=r[0][1].split(',')
            ret[index]=data
            r=rs.fetch_row()
            index+=1
        offset+=block
        done=offset>=count
    return ret
def load_data(file):
    df=np.load(file)
    x_train,y_train,x_test,y_test=df["x_train"],df["y_train"],df["x_test"],df["y_test"]
    return x_train,y_train,x_test,y_test

def shape_data(raw):
    return np.reshape(raw[:,1:],(raw.shape[0],28,28)),raw[:,0]

if __name__ == '__main__':
    filen=filename    
    if len(sys.argv)>1:
        user='martin'
        db='mnist1'
        host='127.0.0.1'
        passwd=''
        try:
            user  = sys.argv[1]
            db    = sys.argv[2]
            host  = sys.argv[3]
            passwd= sys.argv[4]
            filen  = sys.argv[5]
        except:
            pass
        
        db=ms.connect(user=user,db=db,host=host,passwd=passwd)
        
        train=get_table(db,'mnist_train')
        test=get_table(db,'mnist_test')
        for i in range(0,10):
            for j in range(0,28*28+1):
                print train[i,j],' ',
            print 
        x_train,y_train=shape_data(train)
        x_test,y_test=shape_data(test)
        outd={}
        np.savez(filen,x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test)
        print x_train.shape,x_train.shape
    aa=np.load(filen)
    print aa.files
    print aa["x_train"].shape
    print aa["y_train"].shape
    print aa["x_test"].shape
    print aa["y_test"].shape
    aa=np.load("data.npz")
    print aa.files
    print aa["x_train"].shape
    print aa["y_train"].shape
    print aa["x_test"].shape
    print aa["y_test"].shape
    
    
    