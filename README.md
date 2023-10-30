# Week 9: Attention in Sequence-to-Sequence Models

Attention mechanism is a significant enhancement to the traditional sequence-to-sequence models in natural language processing tasks. The term 'attention' refers to the model's ability to focus on specific parts of the input sequence when generating the output sequence, similar to how humans pay attention to different parts of the input when processing information.

How Does Attention Work?
In a standard sequence-to-sequence model with an encoder-decoder architecture, the encoder processes the input sequence into a fixed-size context vector, which is then used by the decoder to generate the output sequence. However, this approach can become a bottleneck, especially for long input sequences, because it requires the model to compress all the necessary information of the input sequence into a single fixed-size vector.

The attention mechanism addresses this limitation by allowing the model to create a shortcut between the encoder and decoder. Instead of condensing all the input information into a single context vector, it lets the decoder look at the entire input sequence directly at each step of output generation.

For each step, the attention mechanism computes a weighted sum of the encoder's outputs, where the weights are determined by how relevant each input element is to the current output step. These weights, also known as attention scores, are calculated using an attention function, which can be learned alongside the rest of the model.

Advantages of Attention Mechanisms
Handling Long Sequences: Attention allows models to handle much longer sequences by alleviating the need for a fixed-size context vector to capture all information in the sequence.

Interpretability: Attention weights can be interpreted as the model's focus, providing insights into what parts of the input sequence the model considers important when generating each part of the output.

Improved Performance: Models with attention often perform better on tasks like machine translation and text summarization compared to their non-attention counterparts.

Example: Machine Translation
In a machine translation task, when generating a translated word, the attention mechanism allows the model to focus on the relevant words in the input sentence. For example, when translating the English sentence "She loves cats" to French "Elle aime les chats", while generating the word "aime" (loves), the model might pay more attention to the input word "loves".

The introduction of attention mechanisms has been a turning point in the development of sequence-to-sequence models, leading to state-of-the-art results in various tasks. Its ability to dynamically weight the importance of input elements brings significant improvements in handling long sequences and enhances the model's interpretability. Notably, the Transformer model, which relies entirely on attention and has no recurrent components, has demonstrated exceptional performance in a variety of language tasks and forms the basis of models like GPT and BERT.

# Readings

[Attention is all you need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

[A Review on Speech Emotion Recognition Using Deep Learning and Attention Mechanism](https://media.proquest.com/media/hms/PFT/1/MBNNJ?_s=gffnUf2uaR6mZK4Gt89qM%2BdO74M%3D)

[Attention mechanisms in computer vision: A survey](https://link.springer.com/article/10.1007/s41095-022-0271-y)

# Extra Readings

[ATTENTION FORCING FOR SEQUENCE-TO-SEQUENCE MODEL TRAINING](https://arxiv.org/pdf/1909.12289.pdf)

[SUPERVISED ATTENTION IN SEQUENCE-TO-SEQUENCE MODELS FOR SPEECH RECOGNITION](https://arxiv.org/pdf/2204.12308.pdf)

# Code example

Step 1: Data Preparation

We start by defining a dummy English to French translation dataset using the TranslationDataset class. This dataset contains training pairs, where each pair consists of an English sentence and its corresponding French translation. We also build vocabularies for the input and output languages to convert words into indices for model processing.

Step 2: Building the Encoder

Next, we define the Encoder class, representing the encoder part of the sequence-to-sequence model. The encoder has an embedding layer to convert input words into dense vectors, and a GRU layer (Gated Recurrent Unit) to process the input sequence. The encoder processes each word in the input sequence and produces both the final hidden state and the entire sequence of hidden states.

Step 3: Building the Attention Layer

We introduce the Attention class, which represents the attention mechanism. Attention allows the decoder to focus on specific parts of the encoder's output at each decoding step. The attention layer calculates attention weights based on the similarity between the decoder's current hidden state and each hidden state in the encoder's output sequence. It then computes a context vector, which is a weighted sum of the encoder's hidden states based on these attention weights.

Step 4: Building the Decoder

We define the Decoder class, representing the decoder part of the sequence-to-sequence model. The decoder has an embedding layer and a GRU layer, similar to the encoder. Additionally, it includes the previously introduced Attention layer. The decoder takes the decoder's input, previous hidden state, and the encoder's output sequence to generate the output probabilities for each word in the output vocabulary.

Step 5: Building the Encoder-Decoder Model with Attention

We create the EncoderDecoderAttention class, combining the encoder and decoder with attention into a single model. During training, this model takes the source sequence (English sentence) and target sequence (French translation) as inputs. The encoder processes the source sequence and produces both the final hidden state and the entire sequence of hidden states. The decoder then uses these hidden states, along with attention, to generate the target sequence. The attention mechanism helps the decoder align with the relevant parts of the source sequence, making the translation more accurate.

Step 6: Training the Model

We define the loss function (CrossEntropyLoss) and the optimizer (Adam) for training. We then run a loop for a specified number of epochs to train the model. In each epoch, we forward propagate the input sequences through the model, calculate the loss by comparing the predicted output with the ground truth output, and update the model's parameters using backpropagation and the Adam optimizer.

Step 7: Translating New Sentences

After training, we define a function (translate_sentence) to translate new English sentences into French. The function takes an English sentence, encodes it using the trained encoder, and then decodes it using the trained decoder with attention. The decoder generates the French translation one word at a time, taking into account the attention mechanism, until it predicts an end-of-sequence token.

Step 8: Test Translation

Finally, we test the trained model by providing a new English sentence ("We are happy") and obtaining its French translation using the translate_sentence function. The translated French sentence is then displayed.
"# research-stay-week-9" 
