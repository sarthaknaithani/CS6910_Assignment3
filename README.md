# CS6910 Deep Learning Assignment 3

## Author: SARTHAK NAITHANI

### Link to the project report:


## Required Libraries
1. `torch`
2. `random`
3. `tqdm` (for progress bar visualization)
4. `numpy`
5. `matplotlib`
6. `pandas`
7. `sklearn.utils.shuffle` (for shuffling dataset)
8. `seaborn` (for attention heatmap visualization)


### Steps to follow:

1. Install wandb by command *pip install wandb* to your system
2. Run train.py file with the appropriate arguments given below
3. Your run will be logged to wandb to my project "CS6910_Assignment3" and entity="cs22m078"
4. You can view the logs that is the model training accuracy, validation accuracy, testing accuracy and respective losses

# File Structre
A breif description of the list of files, and their respective purposes.
1. `CS6910_Assignment3.py` - A class for handling data and all the data related methods
2. `Utils.py` - A genaral set of versatile utility functions that is used in this assignment.
3. `SequenceLearning.py` - This is the main file, containing the Encoder, Decoder, AttentionDecoder, and Seq2Seq model.
4. `BestModels.ipynb` - This file takes care of perfoming the WandB Sweeps for hyper-paramter tuning, and training, testing the best models found in the WandB sweeps.
5. `train.py` - Code to implement the command line interface to interact with the repository

# Classes in Sequence Learning
## `Encoder`
    Init Parameters:
    ----------------
    input_size : english_char_count
    embedding_size : size of each embedding vector
    hidden_size : size of hidden state vector
    num_layers : number of recurrent layers of RNN
    p : dropout probability
    bi_dir : flag to set bidirection in the RNN cell used
    rnn_class: type of RNN to be used in the encoder

    Input:
    ------
    x : torch.Tensor of shape (seq_length, N)
        where seq_length - len of longest string in the batch
        N - batch size

    Output:
    ------
    outputs: torch.Tensor of shape (seq_len, N, hidden_size * D), where D = 2 if bi_dir = True else 1
    hidden: torch.Tensor of shape (num_layers * D, N, hidden_size)

    cell: torch.Tensor of shape (num_layers * D, N, hidden_size) if(rnn_class == "LSTM")
## `Decoder`
    Init Parameters:
    ---------------
    input_size: target_char_count
    embedding_size: size of each embedding vector
    hidden_size: size of hidden state vector
    output_size: number of output features in fully connected layer, here target_char_count
    num_layers : number of recurrent layers of RNN
    p : dropout probability
    bi_dir : flag to set bidirection in the RNN cell used
    rnn_class: type of RNN to be used in the encoder

    Input:
    -----
    x: torch.Tensor of shape (N)
    hidden: torch.Tensor of shape (num_layers * D, N, hidden_size), where D = 2 if bi_dir = True else 1
    cell: torch.Tensor of shape (num_layers * D, N, hidden_size)

    Outputs:
    -------
    predications: torch.Tensor of shape (N, target_char_count)
    hidden: torch.Tensor of shape (num_layers * D, N, hidden_size)
    
    cell: torch.Tensor of shape (num_layers * D, N, hidden_size) if(rnn_class == "LSTM")
## `AttentionDecoder`
    Init Parameters:
    ---------------
    embedding_size: size of each embedding vector
    hidden_size: size of hidden state vector
    output_size: number of output features in fully connected layer, here target_char_count
    num_layers : number of recurrent layers of RNN
    p : dropout probability
    max_length : maximum length of the input word (from encoder) for which this decoder is able to handle - by using attention mechanism
    bi_dir : flag to set bidirection in the RNN cell used
    rnn_class: type of RNN to be used in the encoder

    Input:
    -----
    input: torch.Tensor of shape (N)
    hidden: torch.Tensor of shape (num_layers * D, N, hidden_size), where D = 2 if bi_dir = True else 1
    cell: torch.Tensor of shape (num_layers * D, N, hidden_size) if rnn_class == "LSTM"
    encoder_outputs: torch.Tensor of shape (seq_len, N, hidden_size * D)

    Output:
    ------
    prob : torch.Tensor of shape (N, target_char_count)
    hidden : torch.Tensor of shape (num_layers * D, N, hidden_size), where D = 2 if bi_dir = True else 1
    attn_weights : torch.Tensor of shape (1, N, max_len)

    cell : torch.Tensor of shape (num_layers * D, N, hidden_size) if rnn_class == "LSTM"


# Training
## Steps involved in training
1. Initialize `Encoder`, `Decoder`, `AttnDecoder` and `Seq2Seq` objects
2. Load the data using `get_data()` method
3. Call train_iter() and give appropriate configuration
4. Call `wandb_run_configuration()` method to visualize run in wandb by giving appropriate configuration
5. Visualize the attentions using `plot_attention()` it will download the plots

## TFR method
I emprically observed that the models tend to learn faster if the teacher forcing is performed as a combination of both epoch wise and batch wise.  
I did random teacher forcing by giving teacher ratio of 0.5

# Command Line Interface
Using `train.py` - For help related on how to use `train.py`, type `python train.py -h`  
**Best Models**:  
Configuration for the best model without using attention:  
`python train.py --batch_size 128 --epochs 9 --learning_rate 0.001 --embedding_size 32 --encoder_layers 3 --decoder_layers 2 --enc_dropout 0.3 --dec_dropout 0.3 --hidden_size 256 --rnn_class LSTM --bi_directional`

Configuration for the best model using attention:   
`python train.py --batch_size 128 --epochs 16 --learning_rate 0.001 --embedding_size 128 --encoder_layers 3 --decoder_layers 2 --enc_dropout 0.2 --dec_dropout 0.2 --hidden_size 256 --rnn_class LSTM --bi_directional --attention`
