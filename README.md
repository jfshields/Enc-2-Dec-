# Enc-2-Dec
Deep learning for unsecured lending payment behaviour prediction including RNN-2-MLP Enc-2-Dec, Enc-2-Dec+

Enc-2-Dec+
The Encoder-2-Decoder Plus (Enc-2-Dec+) is a two component sequence-to-sequence Deep Learning model comprised of an ‘encoder’ and a ‘decoder’.

Each component performs a separate task. The encoder is RNN with two recursive elements, as seen in LSTM cells. It is stacked to a depth of five layers. The encoder receives the input sequence and converts it into a fixed length vector. This is passed to the decoder as the input to the first time-step. Additional connections are made with the final hidden and continuous states of the decoder to the first hidden and continuous states of the encoder.

The decoder receives this fixed length vector as the input to the first time-step RNN. The decoder is 20 units wide and stacked by three layers. Each time-step cell is constructed as a LSTM cells. This sequentially generates a predicted value for each time-step. It has an output state which is passed to the above layer and a hidden and a continuous state which to the next time-step. The predicted value is a linearly transformation with a rectified linear activation function of the output of the top layered cell for each time-step.

Enc-2-Dec+ model was designed to allow maximum information to pass from the input sequence to the decoder. Its aim was to enable a lower granularity of patterns from the input sequence to be modelled and represented in the generated predictions. The connectivity between the final hidden and continuous states of the decoder and the first hidden and continuous states of the encoder enable memory to be recalled and utilised in the output sequence. This means longer and more complex temporal behaviours can be modelled and generated. The decoder ensures that each subsequent prediction is conditioned by the previous time-steps prediction as it is the input of that time-step.

The attention connections provide a link from each of the encoder outputs states to each of the first layer of each of the time-steps. This was done by applying a linear transformation with a tanh activation function. This was combined with the input to the each time-step of the bottom layer.

Actual value sampled training was used whereby in the decoder, actual previous values where used as the input to the time-step during training.

A full technical report can be provided on request.
