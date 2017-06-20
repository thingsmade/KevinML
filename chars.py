'''
Created on 18 Jun 2017

@author: martin
'''
import data
import keras.models
import keras.layers
import numpy as np

def model1(input_shape,num_classes):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model
CLASSES=10

def loadWeights(name,model):
    model.load_weights(name)
def saveWeights(name,model):
    model.load_weights(name)
def iterateOnce(model,inp,classes):
    if isinstance(inp,(str,unicode)):
        inp=data.load_data(inp)
    
    trainnr,row,col=inp[0].shape
    testnr,ig1,ig2=inp[2].shape
    if not (ig1==row and ig2==col):
        print 'row and column not matching in test and train'
        raise
    x_train =inp[0].reshape(trainnr, row, col, 1)
    y_train =keras.utils.to_categorical(inp[1],classes)
    x_test=inp[2].reshape(testnr, row, col, 1)
    y_test=keras.utils.to_categorical(inp[3],classes)
#    x_train /= 255
#    x_test /= 255

    input_shape = (row, col, 1)

    m=model1(input_shape,10)
    m.fit(x_train, y_train,
              batch_size=1024,
              epochs=1,
              verbose=1,
              validation_data=(x_test, y_test))
def getConfusion(t1,t2):
    return np.matmul(t1.transpose(),t2/t2.sum(axis=0))
    
    
    
if __name__ == '__main__':
    inp=data.load_data('data.npz')
    trainnr,row,col=inp[0].shape
    testnr,ig1,ig2=inp[2].shape
    if not (ig1==row and ig2==col):
        print 'row and column not matching in test and train'
        exit
    x_train =inp[0].reshape(trainnr, row, col, 1)
    y_train =keras.utils.to_categorical(inp[1],CLASSES)
    x_test=inp[2].reshape(testnr, row, col, 1)
    y_test=keras.utils.to_categorical(inp[3],CLASSES)
    print y_test
#    x_train /= 255
#    x_test /= 255

    input_shape = (row, col, 1)

    if False:
        m=model1(input_shape,10)
        m.fit(x_train, y_train,
                  batch_size=1024,
                  epochs=20,
                  verbose=1,
                  validation_data=(x_test, y_test))
        m.save_weights("play1.wts")
    else:
        m=model1(input_shape,10)
        m.load_weights("play.wts")
        
    r=m.predict(x_test)
    print r.shape,x_test.shape
    print r.sum(axis=1)
    rr=getConfusion(r,y_test)
    for i in range(0,CLASSES):
        for j in range(0,CLASSES):
            print rr[j,i],
        print
    print rr.sum(axis=0)
    print rr.sum(axis=1)
    print r.shape,rr.shape,y_test.shape
    print r[0]
    score = m.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    print(y_test,y_test[0],y_test[0,0])
    print(score)
    max=np.argmax(r, 1)
    zz=np.zeros_like(y_test)
    zz[np.arange(0,10000),max]=1.0
    print zz.shape,max.shape,y_test.shape
    print max
    print zz
    rr=zz*y_test
    print rr.sum(axis=0)/y_test.sum(axis=0)