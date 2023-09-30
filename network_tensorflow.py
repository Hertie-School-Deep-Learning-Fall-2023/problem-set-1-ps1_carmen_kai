import tensorflow as tf
tf.config.run_functions_eagerly(True)



class NeuralNetworkTf(tf.keras.Sequential):

  def __init__(self, sizes, random_state=1):
    
    super().__init__()
    self.sizes = sizes
    self.random_state = random_state
    tf.random.set_seed(random_state)
    
    # error 1: flatten the input data since its a fully connected network
    self.add(tf.keras.layers.Flatten(input_shape=(28,28)))

    for i in range(0, len(sizes)):
      # error 2: correctyed looping condition, use softmax solely at the output layer, use sigmoid for all hidden layers
      if i == len(sizes) - 1:
        self.add(tf.keras.layers.Dense(sizes[i], activation='softmax'))
        
      else:
        self.add(tf.keras.layers.Dense(sizes[i], activation='sigmoid')) # FYI: Using relu here improves the validation accuracy enourmously
        
  
  def compile_and_fit(self, x_train, y_train, 
                      epochs=50, learning_rate=0.01, 
                      batch_size=1,validation_data=None):
    
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    # error 3: used multi-class crossentropy given we are solving a multiclass problem
    loss_function = tf.keras.losses.CategoricalCrossentropy()
    eval_metrics = ['accuracy']

    super().compile(optimizer=optimizer, loss=loss_function, 
                    metrics=eval_metrics)
    return super().fit(x_train, y_train, epochs=epochs, 
                        batch_size=batch_size, 
                        validation_data=validation_data)  



class TimeBasedLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
  '''
  TODO: Implement a time-based learning rate that takes as input a 
  positive integer (initial_learning_rate) and at each step reduces the
  learning rate by 1 until minimal learning rate of 1 is reached.
  '''

  def __init__(self, initial_learning_rate, decay_step=1, min_learning_rate=1):

    super(TimeBasedLearningRate, self).__init__()
    self.initial_learning_rate = initial_learning_rate
    self.decay_step = tf.cast(decay_step, dtype=tf.float32) # cast decay step to float
    self.min_learning_rate = min_learning_rate


  def __call__(self, step):

    step = tf.cast(step, dtype=tf.float32) # ensure the step is float
    learning_rate = tf.maximum(self.min_learning_rate, 
                               self.initial_learning_rate - self.decay_step * step)
    return learning_rate



    