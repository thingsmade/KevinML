'''
Created on 16 Jun 2017

@author: martin
'''
import _mysql as ms
import numpy as np

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
    if True:
        db=ms.connect(user='martin',db='mnist1',host='127.0.0.1',passwd='')
        
    #    train=get_table(db,'mnist_train',count=10,block=2)
        train=get_table(db,'mnist_train')
        test=get_table(db,'mnist_test')
        for i in range(0,10):
            for j in range(0,28*28+1):
                print train[i,j],' ',
            print 
        x_train,y_train=shape_data(train)
        x_test,y_test=shape_data(test)
        outd={}
    #    outd["x_train"]=x_train
    #    outd["y_train"]=y_train
    #    outd["x_test"]=x_test
    #    outd["y_test"]=y_test
        np.savez("dataa.npz",x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test)
        print x_train.shape,x_train.shape
    aa=np.load("mnist.npz")
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
    
    
    