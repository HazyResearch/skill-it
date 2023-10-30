class AbstractTrainer():
    
    def __init__(self):
        pass
     
    def train(self,
              args,
              logger,
              tokenizer,
              model,
              validation_data,
              evaluator,
              ):
        raise NotImplementedError