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
    filen=data.filen
    if len(sys.argv)>1:
        pass
    inp=data.load_data(filen)
    trainnr,row,col=inp[0].shape
    testnr,ig1,ig2=inp[2].shape
    if not (ig1==row and ig2==col):
        print 'row and column not matching in test and train'
        exit
    x_train =inp[0].reshape(trainnr, row, col, 1)
    y_train =keras.utils.to_categorical(inp[1],CLASSES)
    x_test=inp[2].reshape(testnr, row, col, 1)
    y_test=keras.utils.to_categorical(inp[3],CLASSES)

    input_shape = (row, col, 1)

    if len(sys.argv)==1:
        m=model1(input_shape,10)
        m.fit(x_train, y_train,
                  batch_size=1024,
                  epochs=20,
                  verbose=1,
                  validation_data=(x_test, y_test))
        m.save_weights("model.wts")
    else:
        
        m=model1(input_shape,10)
        m.load_weights("model.wts")
        
    r=m.predict(x_test)
    rr=getConfusion(r,y_test)
    print 'Confusion matrix'
    for i in range(0,CLASSES):
        print "class %i :"%i,
        for j in range(0,CLASSES):
            print rr[j,i],
        print
    score = m.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    max=np.argmax(r, 1)
    zz=np.zeros_like(y_test)
    zz[np.arange(0,10000),max]=1.0
    rr=zz*y_test
    print "classification accuracy per character"
    print rr.sum(axis=0)/y_test.sum(axis=0)
    print "classification accuracy"
    print rr.sum()/y_test.sum()