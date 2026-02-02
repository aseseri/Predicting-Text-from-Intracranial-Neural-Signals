
modelName = 'modelNameHere'

args = {}
args['outputDir'] = 'logs/' + modelName
args['checkpointPath'] = 'logs/' + modelName + '/modelWeights'
args['datasetPath'] = 'data/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 128 # baseline: 64, exp 12: 128
args['lrStart'] = 0.02
args['lrEnd'] = 0.02
args['nUnits'] = 1024 # baseline: 1024, exp 5: 2048
args['nBatch'] = 10000 # baseline: 10000
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.5 # baseline: 0.4, exp 5: 0.5, exp 14: 0.5
args['whiteNoiseSD'] = 1.0 # baseline: 0.8, exp 13: 1.0
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = False
args['l2_decay'] = 1e-5

# Transformer Param
args['Transformer-hidden-nLayers'] = 512 # exp 3/4: 256, exp 6: 128
args['Transformer-nLayers'] = 4 # exp 3/4: 4, exp 6: 2
args['Transformer-dropout'] = 0.1 # exp 3/4: 0.1, exp 6: 0.3

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
